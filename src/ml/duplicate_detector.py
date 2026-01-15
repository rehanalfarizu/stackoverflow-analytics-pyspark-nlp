"""
Duplicate Detector Module
=========================
Deteksi pertanyaan duplikat di Stack Overflow.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, ArrayType, StringType
from pyspark.ml.feature import Word2Vec, Word2VecModel, HashingTF, MinHashLSH
from pyspark.ml.linalg import Vectors
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """
    Detector untuk pertanyaan duplikat.
    
    Menggunakan:
    - Jaccard Similarity
    - Cosine Similarity dengan TF-IDF
    - Word2Vec Embeddings
    - MinHash LSH untuk scalability
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize Duplicate Detector.
        
        Parameters
        ----------
        spark : SparkSession
            Active Spark session
        """
        self.spark = spark
        self.word2vec_model = None
        self.minhash_model = None
    
    def calculate_jaccard_similarity(
        self,
        df: DataFrame,
        tokens_column: str = "ProcessedText"
    ) -> DataFrame:
        """
        Menghitung Jaccard similarity antar dokumen menggunakan MinHash LSH.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame dengan tokens
        tokens_column : str
            Kolom tokens
            
        Returns
        -------
        DataFrame
            DataFrame dengan kolom untuk similarity lookup
        """
        logger.info("Setting up Jaccard similarity with MinHash LSH")
        
        # Hash tokens to feature vector
        hashingTF = HashingTF(
            inputCol=tokens_column,
            outputCol="features",
            numFeatures=10000
        )
        df_hashed = hashingTF.transform(df)
        
        # Fit MinHash LSH
        minhash = MinHashLSH(
            inputCol="features",
            outputCol="hashes",
            numHashTables=5
        )
        
        self.minhash_model = minhash.fit(df_hashed)
        df_transformed = self.minhash_model.transform(df_hashed)
        
        return df_transformed
    
    def find_similar_questions(
        self,
        df: DataFrame,
        query_df: DataFrame,
        threshold: float = 0.5,
        num_results: int = 10
    ) -> DataFrame:
        """
        Mencari pertanyaan serupa untuk query.
        
        Parameters
        ----------
        df : DataFrame
            Corpus pertanyaan (sudah di-transform)
        query_df : DataFrame
            Pertanyaan query (sudah di-transform)
        threshold : float
            Threshold similarity (0-1, untuk Jaccard distance ini 1-similarity)
        num_results : int
            Jumlah hasil maksimum
            
        Returns
        -------
        DataFrame
            Pertanyaan serupa
        """
        if self.minhash_model is None:
            raise ValueError("MinHash model not fitted. Call calculate_jaccard_similarity first.")
        
        # Get first query
        query_row = query_df.first()
        query_key = query_row["features"]
        
        # Approximate similarity search
        similar = self.minhash_model.approxNearestNeighbors(
            df, query_key, numNearestNeighbors=num_results
        )
        
        return similar
    
    def train_word2vec(
        self,
        df: DataFrame,
        tokens_column: str = "ProcessedText",
        vector_size: int = 100,
        min_count: int = 5
    ) -> None:
        """
        Melatih Word2Vec untuk similarity.
        
        Parameters
        ----------
        df : DataFrame
            Training data
        tokens_column : str
            Kolom tokens
        vector_size : int
            Dimensi vektor
        min_count : int
            Minimum word count
        """
        logger.info("Training Word2Vec model")
        
        word2vec = Word2Vec(
            inputCol=tokens_column,
            outputCol="word2vec_features",
            vectorSize=vector_size,
            minCount=min_count,
            maxIter=10
        )
        
        self.word2vec_model = word2vec.fit(df)
        logger.info("Word2Vec training completed")
    
    def get_document_vectors(
        self,
        df: DataFrame,
        tokens_column: str = "ProcessedText"
    ) -> DataFrame:
        """
        Mendapatkan vektor dokumen menggunakan Word2Vec.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame dengan tokens
        tokens_column : str
            Kolom tokens
            
        Returns
        -------
        DataFrame
            DataFrame dengan word2vec_features
        """
        if self.word2vec_model is None:
            self.train_word2vec(df, tokens_column)
        
        return self.word2vec_model.transform(df)
    
    def calculate_cosine_similarity(
        self,
        vec1,
        vec2
    ) -> float:
        """
        Menghitung cosine similarity antara dua vektor.
        
        Parameters
        ----------
        vec1 : Vector
            Vektor pertama
        vec2 : Vector
            Vektor kedua
            
        Returns
        -------
        float
            Cosine similarity (-1 to 1)
        """
        import numpy as np
        
        a = np.array(vec1.toArray())
        b = np.array(vec2.toArray())
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def find_duplicates_batch(
        self,
        df: DataFrame,
        threshold: float = 0.85
    ) -> DataFrame:
        """
        Mencari semua pasangan duplikat dalam batch.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame dengan vektor fitur
        threshold : float
            Threshold similarity
            
        Returns
        -------
        DataFrame
            Pasangan duplikat
        """
        logger.info(f"Finding duplicate pairs with threshold {threshold}")
        
        if self.minhash_model is None:
            df = self.calculate_jaccard_similarity(df)
        
        # Approximate similarity join
        duplicates = self.minhash_model.approxSimilarityJoin(
            df, df, threshold=1.0 - threshold,  # Jaccard distance
            distCol="JaccardDistance"
        )
        
        # Filter self-matches dan duplikasi
        duplicates = duplicates \
            .filter(F.col("datasetA.Id") < F.col("datasetB.Id")) \
            .withColumn("Similarity", 1 - F.col("JaccardDistance"))
        
        return duplicates
    
    def detect_near_duplicates(
        self,
        df: DataFrame,
        text_column: str = "Title",
        threshold: float = 0.9
    ) -> DataFrame:
        """
        Mendeteksi near-duplicate berdasarkan title.
        
        Parameters
        ----------
        df : DataFrame
            Posts DataFrame
        text_column : str
            Kolom teks untuk perbandingan
        threshold : float
            Threshold similarity
            
        Returns
        -------
        DataFrame
            Near-duplicate pairs
        """
        logger.info("Detecting near-duplicate questions")
        
        # Simple character-level similarity using n-grams
        @F.udf(returnType=ArrayType(StringType()))
        def get_character_ngrams(text, n=3):
            if text is None:
                return []
            text = text.lower()
            return [text[i:i+n] for i in range(len(text) - n + 1)]
        
        # Add n-grams
        df_ngrams = df.withColumn("CharNgrams", get_character_ngrams(F.col(text_column)))
        
        # Use MinHash for scalability
        hashingTF = HashingTF(
            inputCol="CharNgrams",
            outputCol="ngram_features",
            numFeatures=5000
        )
        df_hashed = hashingTF.transform(df_ngrams)
        
        minhash = MinHashLSH(
            inputCol="ngram_features",
            outputCol="ngram_hashes",
            numHashTables=3
        )
        model = minhash.fit(df_hashed)
        df_transformed = model.transform(df_hashed)
        
        # Find similar pairs
        duplicates = model.approxSimilarityJoin(
            df_transformed, df_transformed,
            threshold=1.0 - threshold,
            distCol="Distance"
        )
        
        # Clean up
        duplicates = duplicates \
            .filter(F.col("datasetA.Id") < F.col("datasetB.Id")) \
            .select(
                F.col("datasetA.Id").alias("QuestionId1"),
                F.col("datasetA.Title").alias("Title1"),
                F.col("datasetB.Id").alias("QuestionId2"),
                F.col("datasetB.Title").alias("Title2"),
                (1 - F.col("Distance")).alias("Similarity")
            ) \
            .orderBy(F.desc("Similarity"))
        
        return duplicates
    
    def get_similar_by_id(
        self,
        df: DataFrame,
        question_id: int,
        top_k: int = 10
    ) -> DataFrame:
        """
        Mendapatkan pertanyaan serupa untuk ID tertentu.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame dengan fitur
        question_id : int
            ID pertanyaan
        top_k : int
            Jumlah hasil
            
        Returns
        -------
        DataFrame
            Pertanyaan serupa
        """
        # Get the question
        question = df.filter(F.col("Id") == question_id)
        
        if question.count() == 0:
            logger.warning(f"Question {question_id} not found")
            return self.spark.createDataFrame([], df.schema)
        
        # Find similar
        return self.find_similar_questions(df, question, num_results=top_k)
    
    def evaluate_duplicate_detection(
        self,
        predictions_df: DataFrame,
        ground_truth_df: DataFrame
    ) -> dict:
        """
        Evaluasi performa duplicate detection.
        
        Parameters
        ----------
        predictions_df : DataFrame
            Prediksi duplikat
        ground_truth_df : DataFrame
            Ground truth (PostLinks dengan LinkTypeId=3)
            
        Returns
        -------
        dict
            Metrics (precision, recall, F1)
        """
        # Join predictions with ground truth
        # Assuming ground_truth has PostId and RelatedPostId columns
        
        # True positives: predictions yang ada di ground truth
        tp = predictions_df.join(
            ground_truth_df,
            (predictions_df.QuestionId1 == ground_truth_df.PostId) &
            (predictions_df.QuestionId2 == ground_truth_df.RelatedPostId),
            "inner"
        ).count()
        
        # False positives: predictions yang tidak ada di ground truth
        fp = predictions_df.count() - tp
        
        # False negatives: ground truth yang tidak diprediksi
        fn = ground_truth_df.join(
            predictions_df,
            (ground_truth_df.PostId == predictions_df.QuestionId1) &
            (ground_truth_df.RelatedPostId == predictions_df.QuestionId2),
            "left_anti"
        ).count()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def save_model(self, path: str) -> None:
        """Menyimpan model."""
        if self.word2vec_model:
            self.word2vec_model.save(f"{path}/word2vec")
        if self.minhash_model:
            self.minhash_model.save(f"{path}/minhash")
        logger.info(f"Models saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Memuat model yang tersimpan."""
        try:
            self.word2vec_model = Word2VecModel.load(f"{path}/word2vec")
        except Exception:
            logger.warning("Word2Vec model not found")
        
        try:
            from pyspark.ml.feature import MinHashLSHModel
            self.minhash_model = MinHashLSHModel.load(f"{path}/minhash")
        except Exception:
            logger.warning("MinHash model not found")
