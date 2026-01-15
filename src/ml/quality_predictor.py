"""
Quality Predictor Module
========================
Prediksi kualitas pertanyaan Stack Overflow.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import (
    RandomForestClassifier, GBTClassifier, 
    LogisticRegression, MultilayerPerceptronClassifier
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class QualityPredictor:
    """
    Predictor untuk kualitas pertanyaan Stack Overflow.
    
    Memprediksi apakah pertanyaan akan:
    - Mendapat skor tinggi
    - Mendapat jawaban yang diterima
    - Mendapat banyak views
    """
    
    # Feature columns
    DEFAULT_FEATURES = [
        'TitleLength', 'TitleWordCount', 'BodyLength', 'BodyWordCount',
        'NumTags', 'HasCode', 'CodeRatio', 'HasQuestionMark',
        'NumQuestionMarks', 'PostHour', 'PostDayOfWeek', 'IsWeekend'
    ]
    
    def __init__(self, spark: SparkSession):
        """
        Initialize Quality Predictor.
        
        Parameters
        ----------
        spark : SparkSession
            Active Spark session
        """
        self.spark = spark
        self.model = None
        self.pipeline = None
        self.feature_columns = self.DEFAULT_FEATURES.copy()
    
    def prepare_features(
        self,
        df: DataFrame,
        label_column: str = "QualityLabel",
        feature_columns: List[str] = None
    ) -> DataFrame:
        """
        Menyiapkan fitur untuk training.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame
        label_column : str
            Kolom label
        feature_columns : List[str], optional
            Kolom fitur
            
        Returns
        -------
        DataFrame
            DataFrame dengan features vector
        """
        if feature_columns:
            self.feature_columns = feature_columns
        
        # Filter untuk kolom yang ada
        available_features = [c for c in self.feature_columns if c in df.columns]
        
        if len(available_features) < len(self.feature_columns):
            missing = set(self.feature_columns) - set(available_features)
            logger.warning(f"Missing features: {missing}")
        
        # Fill nulls untuk numeric columns
        for col in available_features:
            df = df.fillna({col: 0})
        
        # Vector assembler
        assembler = VectorAssembler(
            inputCols=available_features,
            outputCol="features",
            handleInvalid="skip"
        )
        
        df = assembler.transform(df)
        
        # Rename label column
        if label_column != "label":
            df = df.withColumn("label", F.col(label_column).cast("double"))
        
        self.feature_columns = available_features
        return df
    
    def train(
        self,
        df: DataFrame,
        algorithm: str = "random_forest",
        label_column: str = "QualityLabel",
        feature_columns: List[str] = None,
        test_size: float = 0.2
    ) -> Dict:
        """
        Melatih model prediksi kualitas.
        
        Parameters
        ----------
        df : DataFrame
            Training data
        algorithm : str
            Algoritma: random_forest, gradient_boosting, logistic
        label_column : str
            Kolom label
        feature_columns : List[str], optional
            Kolom fitur
        test_size : float
            Proporsi test set
            
        Returns
        -------
        Dict
            Training metrics
        """
        logger.info(f"Training quality predictor with {algorithm}")
        
        # Prepare features
        df = self.prepare_features(df, label_column, feature_columns)
        
        # Split data
        train_df, test_df = df.randomSplit([1 - test_size, test_size], seed=42)
        
        logger.info(f"Training set size: {train_df.count()}")
        logger.info(f"Test set size: {test_df.count()}")
        
        # Select classifier
        if algorithm == "random_forest":
            classifier = RandomForestClassifier(
                featuresCol="features",
                labelCol="label",
                numTrees=100,
                maxDepth=10,
                seed=42
            )
        elif algorithm == "gradient_boosting":
            classifier = GBTClassifier(
                featuresCol="features",
                labelCol="label",
                maxIter=100,
                maxDepth=10,
                seed=42
            )
        elif algorithm == "logistic":
            classifier = LogisticRegression(
                featuresCol="features",
                labelCol="label",
                maxIter=100
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Train model
        self.model = classifier.fit(train_df)
        
        # Evaluate
        predictions = self.model.transform(test_df)
        metrics = self._evaluate(predictions)
        
        logger.info(f"Training completed. Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def train_with_cv(
        self,
        df: DataFrame,
        algorithm: str = "random_forest",
        label_column: str = "QualityLabel",
        num_folds: int = 5
    ) -> Dict:
        """
        Training dengan Cross Validation.
        
        Parameters
        ----------
        df : DataFrame
            Training data
        algorithm : str
            Algoritma
        label_column : str
            Kolom label
        num_folds : int
            Jumlah folds
            
        Returns
        -------
        Dict
            Best metrics
        """
        logger.info(f"Training with {num_folds}-fold cross validation")
        
        # Prepare features
        df = self.prepare_features(df, label_column)
        
        # Define classifier and param grid
        if algorithm == "random_forest":
            classifier = RandomForestClassifier(
                featuresCol="features",
                labelCol="label",
                seed=42
            )
            param_grid = ParamGridBuilder() \
                .addGrid(classifier.numTrees, [50, 100, 200]) \
                .addGrid(classifier.maxDepth, [5, 10, 15]) \
                .build()
        else:
            classifier = GBTClassifier(
                featuresCol="features",
                labelCol="label",
                seed=42
            )
            param_grid = ParamGridBuilder() \
                .addGrid(classifier.maxIter, [50, 100]) \
                .addGrid(classifier.maxDepth, [5, 10]) \
                .build()
        
        # Evaluator
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        
        # Cross validator
        cv = CrossValidator(
            estimator=classifier,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=num_folds,
            seed=42
        )
        
        # Fit
        cv_model = cv.fit(df)
        self.model = cv_model.bestModel
        
        # Get best metrics
        predictions = self.model.transform(df)
        metrics = self._evaluate(predictions)
        
        logger.info(f"Best accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def _evaluate(self, predictions: DataFrame) -> Dict:
        """Evaluasi model."""
        
        # Multiclass evaluator
        mc_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction"
        )
        
        accuracy = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "accuracy"})
        f1 = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "f1"})
        precision = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "weightedPrecision"})
        recall = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "weightedRecall"})
        
        # Binary evaluator for AUC (if binary)
        try:
            bc_evaluator = BinaryClassificationEvaluator(
                labelCol="label",
                rawPredictionCol="rawPrediction"
            )
            auc = bc_evaluator.evaluate(predictions)
        except Exception:
            auc = None
        
        metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        
        return metrics
    
    def predict(
        self,
        df: DataFrame,
        include_probability: bool = True
    ) -> DataFrame:
        """
        Prediksi kualitas pertanyaan.
        
        Parameters
        ----------
        df : DataFrame
            Data untuk prediksi
        include_probability : bool
            Sertakan probabilitas
            
        Returns
        -------
        DataFrame
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare features
        df = self.prepare_features(df, "QualityLabel")
        
        # Predict
        predictions = self.model.transform(df)
        
        # Select relevant columns
        output_cols = ["Id", "Title", "prediction"]
        if include_probability and "probability" in predictions.columns:
            output_cols.append("probability")
        
        return predictions.select(*output_cols)
    
    def get_feature_importance(self) -> List[Tuple[str, float]]:
        """
        Mendapatkan feature importance.
        
        Returns
        -------
        List[Tuple[str, float]]
            List of (feature_name, importance)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if hasattr(self.model, 'featureImportances'):
            importances = self.model.featureImportances.toArray()
            
            feature_importance = list(zip(self.feature_columns, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            return feature_importance
        else:
            logger.warning("Model does not support feature importance")
            return []
    
    def print_feature_importance(self) -> None:
        """Print feature importance dalam format yang mudah dibaca."""
        
        importances = self.get_feature_importance()
        
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE")
        print("="*50)
        
        for feature, importance in importances:
            bar = "#" * int(importance * 50)
            print(f"{feature:25s} {importance:.4f} {bar}")
        
        print("="*50)
    
    def save_model(self, path: str) -> None:
        """
        Menyimpan model.
        
        Parameters
        ----------
        path : str
            Path untuk menyimpan
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str, algorithm: str = "random_forest") -> None:
        """
        Memuat model yang tersimpan.
        
        Parameters
        ----------
        path : str
            Path model
        algorithm : str
            Tipe algoritma
        """
        if algorithm == "random_forest":
            from pyspark.ml.classification import RandomForestClassificationModel
            self.model = RandomForestClassificationModel.load(path)
        elif algorithm == "gradient_boosting":
            from pyspark.ml.classification import GBTClassificationModel
            self.model = GBTClassificationModel.load(path)
        else:
            from pyspark.ml.classification import LogisticRegressionModel
            self.model = LogisticRegressionModel.load(path)
        
        logger.info(f"Model loaded from {path}")


def create_quality_labels(
    df: DataFrame,
    score_threshold: int = 5,
    view_threshold: int = 1000
) -> DataFrame:
    """
    Membuat label kualitas untuk training.
    
    Parameters
    ----------
    df : DataFrame
        Posts DataFrame
    score_threshold : int
        Threshold skor
    view_threshold : int
        Threshold views
        
    Returns
    -------
    DataFrame
        DataFrame dengan QualityLabel
    """
    return df.withColumn(
        "QualityLabel",
        F.when(
            (F.col("Score") >= score_threshold) & 
            (F.col("ViewCount") >= view_threshold) &
            (F.col("HasAcceptedAnswer") == 1),
            2  # High quality
        ).when(
            (F.col("Score") >= 0) & (F.col("AnswerCount") > 0),
            1  # Medium quality
        ).otherwise(0)  # Low quality
    )
