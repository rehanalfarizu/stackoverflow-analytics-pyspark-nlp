"""
Feature Extractor Module
========================
Ekstraksi fitur untuk NLP dan Machine Learning.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import (
    HashingTF, IDF, CountVectorizer, Word2Vec,
    VectorAssembler, StandardScaler, MinMaxScaler
)
from pyspark.ml.linalg import Vectors, VectorUDT
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Ekstraksi fitur dari teks untuk machine learning.
    
    Mendukung:
    - TF-IDF
    - Word2Vec
    - Count Vectorizer
    - Feature engineering
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize Feature Extractor.
        
        Parameters
        ----------
        spark : SparkSession
            Active Spark session
        """
        self.spark = spark
        self.fitted_models = {}
    
    def extract_tfidf(
        self,
        df: DataFrame,
        input_column: str = "ProcessedText",
        output_column: str = "TfidfFeatures",
        num_features: int = 10000,
        min_doc_freq: int = 5
    ) -> Tuple[DataFrame, dict]:
        """
        Ekstrak fitur TF-IDF.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame dengan tokens
        input_column : str
            Kolom tokens
        output_column : str
            Kolom output TF-IDF
        num_features : int
            Jumlah fitur
        min_doc_freq : int
            Minimum document frequency
            
        Returns
        -------
        Tuple[DataFrame, dict]
            (DataFrame dengan TF-IDF, fitted models)
        """
        logger.info(f"Extracting TF-IDF features with {num_features} features")
        
        # Hashing TF
        tf_col = f"{output_column}_tf"
        hashing_tf = HashingTF(
            inputCol=input_column,
            outputCol=tf_col,
            numFeatures=num_features
        )
        df = hashing_tf.transform(df)
        
        # IDF
        idf = IDF(inputCol=tf_col, outputCol=output_column, minDocFreq=min_doc_freq)
        idf_model = idf.fit(df)
        df = idf_model.transform(df)
        
        # Clean up
        df = df.drop(tf_col)
        
        self.fitted_models['tfidf'] = {
            'hashing_tf': hashing_tf,
            'idf_model': idf_model
        }
        
        logger.info("TF-IDF extraction completed")
        return df, self.fitted_models['tfidf']
    
    def extract_count_vectors(
        self,
        df: DataFrame,
        input_column: str = "ProcessedText",
        output_column: str = "CountFeatures",
        vocab_size: int = 10000,
        min_doc_freq: int = 5,
        max_doc_freq: float = 0.95
    ) -> Tuple[DataFrame, object]:
        """
        Ekstrak fitur menggunakan Count Vectorizer.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame
        input_column : str
            Kolom tokens
        output_column : str
            Kolom output
        vocab_size : int
            Ukuran vocabulary
        min_doc_freq : int
            Minimum document frequency
        max_doc_freq : float
            Maximum document frequency ratio
            
        Returns
        -------
        Tuple[DataFrame, object]
            (DataFrame dengan count vectors, fitted model)
        """
        logger.info(f"Extracting count vectors with vocab size {vocab_size}")
        
        cv = CountVectorizer(
            inputCol=input_column,
            outputCol=output_column,
            vocabSize=vocab_size,
            minDF=min_doc_freq,
            maxDF=max_doc_freq
        )
        
        cv_model = cv.fit(df)
        df = cv_model.transform(df)
        
        self.fitted_models['count_vectorizer'] = cv_model
        
        # Get vocabulary
        vocabulary = cv_model.vocabulary
        logger.info(f"Vocabulary size: {len(vocabulary)}")
        
        return df, cv_model
    
    def extract_word2vec(
        self,
        df: DataFrame,
        input_column: str = "ProcessedText",
        output_column: str = "Word2VecFeatures",
        vector_size: int = 100,
        window_size: int = 5,
        min_count: int = 5,
        max_iter: int = 10
    ) -> Tuple[DataFrame, object]:
        """
        Ekstrak fitur menggunakan Word2Vec.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame
        input_column : str
            Kolom tokens
        output_column : str
            Kolom output
        vector_size : int
            Dimensi vektor
        window_size : int
            Window size untuk konteks
        min_count : int
            Minimum word count
        max_iter : int
            Jumlah iterasi training
            
        Returns
        -------
        Tuple[DataFrame, object]
            (DataFrame dengan Word2Vec vectors, fitted model)
        """
        logger.info(f"Training Word2Vec with vector size {vector_size}")
        
        word2vec = Word2Vec(
            inputCol=input_column,
            outputCol=output_column,
            vectorSize=vector_size,
            windowSize=window_size,
            minCount=min_count,
            maxIter=max_iter
        )
        
        w2v_model = word2vec.fit(df)
        df = w2v_model.transform(df)
        
        self.fitted_models['word2vec'] = w2v_model
        
        logger.info("Word2Vec training completed")
        return df, w2v_model
    
    def get_similar_words(
        self,
        word: str,
        n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Mendapatkan kata-kata yang mirip menggunakan Word2Vec.
        
        Parameters
        ----------
        word : str
            Kata untuk dicari sinonim
        n : int
            Jumlah kata mirip
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (word, similarity) tuples
        """
        if 'word2vec' not in self.fitted_models:
            raise ValueError("Word2Vec model not fitted yet")
        
        model = self.fitted_models['word2vec']
        return model.findSynonyms(word, n).collect()
    
    def extract_numeric_features(
        self,
        df: DataFrame,
        feature_columns: List[str],
        output_column: str = "NumericFeatures"
    ) -> DataFrame:
        """
        Mengekstrak dan menggabungkan fitur numerik.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame
        feature_columns : List[str]
            Kolom fitur numerik
        output_column : str
            Nama kolom output vector
            
        Returns
        -------
        DataFrame
            DataFrame dengan vektor fitur
        """
        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol=output_column,
            handleInvalid="skip"
        )
        
        return assembler.transform(df)
    
    def scale_features(
        self,
        df: DataFrame,
        input_column: str,
        output_column: str = None,
        method: str = "standard"
    ) -> Tuple[DataFrame, object]:
        """
        Scaling fitur numerik.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame
        input_column : str
            Kolom fitur vektor
        output_column : str, optional
            Kolom output (default: input_column + "_scaled")
        method : str
            Metode scaling: "standard" atau "minmax"
            
        Returns
        -------
        Tuple[DataFrame, object]
            (Scaled DataFrame, scaler model)
        """
        if output_column is None:
            output_column = f"{input_column}_scaled"
        
        if method == "standard":
            scaler = StandardScaler(
                inputCol=input_column,
                outputCol=output_column,
                withStd=True,
                withMean=True
            )
        elif method == "minmax":
            scaler = MinMaxScaler(
                inputCol=input_column,
                outputCol=output_column
            )
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        scaler_model = scaler.fit(df)
        df = scaler_model.transform(df)
        
        return df, scaler_model
    
    def extract_question_features(self, df: DataFrame) -> DataFrame:
        """
        Ekstrak fitur khusus untuk pertanyaan Stack Overflow.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame dengan kolom pertanyaan
            
        Returns
        -------
        DataFrame
            DataFrame dengan fitur tambahan
        """
        logger.info("Extracting question-specific features")
        
        # Text-based features
        df = df \
            .withColumn("TitleLength", F.length("Title")) \
            .withColumn("BodyLength", F.length(F.coalesce("BodyClean", F.lit("")))) \
            .withColumn("TitleWordCount", 
                F.size(F.split("Title", r"\s+"))) \
            .withColumn("BodyWordCount",
                F.size(F.split(F.coalesce("BodyClean", F.lit("")), r"\s+")))
        
        # Question indicators
        df = df \
            .withColumn("HasQuestionMark",
                F.when(F.col("Title").contains("?"), 1).otherwise(0)) \
            .withColumn("NumQuestionMarks",
                F.length(F.regexp_replace("Title", r"[^?]", "")))
        
        # Code indicators
        df = df \
            .withColumn("CodeRatio",
                F.length(F.coalesce("CodeBlocks", F.lit(""))) / 
                F.greatest(F.col("BodyLength"), F.lit(1)))
        
        # Tag-based features
        df = df \
            .withColumn("NumTags", 
                F.coalesce(F.size("TagsList"), F.lit(0)))
        
        # Time-based features
        df = df \
            .withColumn("PostHour", F.hour("CreationDate")) \
            .withColumn("PostDayOfWeek", F.dayofweek("CreationDate")) \
            .withColumn("IsWeekend",
                F.when(F.col("PostDayOfWeek").isin([1, 7]), 1).otherwise(0))
        
        return df
    
    def combine_features(
        self,
        df: DataFrame,
        text_feature_col: str,
        numeric_feature_cols: List[str],
        output_column: str = "CombinedFeatures"
    ) -> DataFrame:
        """
        Menggabungkan fitur teks dan numerik.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame
        text_feature_col : str
            Kolom fitur teks (vector)
        numeric_feature_cols : List[str]
            Kolom fitur numerik
        output_column : str
            Kolom output
            
        Returns
        -------
        DataFrame
            DataFrame dengan fitur gabungan
        """
        # First combine numeric features
        numeric_output = "temp_numeric_features"
        df = self.extract_numeric_features(
            df, 
            numeric_feature_cols, 
            numeric_output
        )
        
        # Then combine with text features
        assembler = VectorAssembler(
            inputCols=[text_feature_col, numeric_output],
            outputCol=output_column,
            handleInvalid="skip"
        )
        
        df = assembler.transform(df)
        
        # Clean up
        df = df.drop(numeric_output)
        
        return df
