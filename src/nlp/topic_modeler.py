"""
Topic Modeler Module
====================
Topic modeling untuk analisis konten Stack Overflow.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.clustering import LDA, BisectingKMeans
from pyspark.ml.feature import CountVectorizer, IDF
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TopicModeler:
    """
    Topic Modeling menggunakan LDA dan clustering.
    
    Mengidentifikasi topik-topik tersembunyi dalam
    corpus pertanyaan Stack Overflow.
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize Topic Modeler.
        
        Parameters
        ----------
        spark : SparkSession
            Active Spark session
        """
        self.spark = spark
        self.lda_model = None
        self.cv_model = None
        self.vocabulary = None
    
    def fit_lda(
        self,
        df: DataFrame,
        input_column: str = "ProcessedText",
        num_topics: int = 20,
        max_iterations: int = 100,
        vocab_size: int = 10000,
        min_doc_freq: int = 5
    ) -> Tuple[object, DataFrame]:
        """
        Fit LDA topic model.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame dengan tokens
        input_column : str
            Kolom tokens
        num_topics : int
            Jumlah topik
        max_iterations : int
            Maksimum iterasi
        vocab_size : int
            Ukuran vocabulary
        min_doc_freq : int
            Minimum document frequency
            
        Returns
        -------
        Tuple[object, DataFrame]
            (LDA model, DataFrame dengan topik)
        """
        logger.info(f"Fitting LDA with {num_topics} topics")
        
        # Vectorize text
        cv = CountVectorizer(
            inputCol=input_column,
            outputCol="cv_features",
            vocabSize=vocab_size,
            minDF=min_doc_freq
        )
        self.cv_model = cv.fit(df)
        df_vectorized = self.cv_model.transform(df)
        self.vocabulary = self.cv_model.vocabulary
        
        # Fit LDA
        lda = LDA(
            k=num_topics,
            maxIter=max_iterations,
            featuresCol="cv_features",
            topicDistributionCol="TopicDistribution",
            optimizer="online",
            learningOffset=1024.0,
            learningDecay=0.51
        )
        
        self.lda_model = lda.fit(df_vectorized)
        
        # Transform to get topic distributions
        df_topics = self.lda_model.transform(df_vectorized)
        
        # Get dominant topic
        @F.udf("integer")
        def get_dominant_topic(distribution):
            if distribution is None:
                return -1
            return int(distribution.argmax())
        
        df_topics = df_topics.withColumn(
            "DominantTopic", 
            get_dominant_topic("TopicDistribution")
        )
        
        # Log model metrics
        log_likelihood = self.lda_model.logLikelihood(df_vectorized)
        log_perplexity = self.lda_model.logPerplexity(df_vectorized)
        logger.info(f"Log Likelihood: {log_likelihood}")
        logger.info(f"Log Perplexity: {log_perplexity}")
        
        return self.lda_model, df_topics
    
    def get_topics(
        self,
        num_words: int = 10
    ) -> List[Dict]:
        """
        Mendapatkan topik dengan kata-kata representatif.
        
        Parameters
        ----------
        num_words : int
            Jumlah kata per topik
            
        Returns
        -------
        List[Dict]
            List topik dengan kata-kata dan bobotnya
        """
        if self.lda_model is None or self.vocabulary is None:
            raise ValueError("LDA model not fitted yet")
        
        topics = []
        
        # Get topic-word matrix
        topic_indices = self.lda_model.describeTopics(maxTermsPerTopic=num_words)
        
        for row in topic_indices.collect():
            topic_id = row['topic']
            term_indices = row['termIndices']
            term_weights = row['termWeights']
            
            words = []
            for idx, weight in zip(term_indices, term_weights):
                if idx < len(self.vocabulary):
                    words.append({
                        'word': self.vocabulary[idx],
                        'weight': float(weight)
                    })
            
            topics.append({
                'topic_id': topic_id,
                'words': words
            })
        
        return topics
    
    def print_topics(self, num_words: int = 10) -> None:
        """
        Print topik dalam format yang mudah dibaca.
        
        Parameters
        ----------
        num_words : int
            Jumlah kata per topik
        """
        topics = self.get_topics(num_words)
        
        print("\n" + "="*60)
        print("DISCOVERED TOPICS")
        print("="*60)
        
        for topic in topics:
            words_str = ", ".join([
                f"{w['word']} ({w['weight']:.3f})" 
                for w in topic['words']
            ])
            print(f"\nTopic {topic['topic_id']}:")
            print(f"  {words_str}")
        
        print("\n" + "="*60)
    
    def get_topic_distribution(
        self,
        df: DataFrame
    ) -> DataFrame:
        """
        Mendapatkan distribusi topik per dokumen.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame dengan TopicDistribution
            
        Returns
        -------
        DataFrame
            Agregasi distribusi topik
        """
        # Count documents per dominant topic
        topic_counts = df.groupBy("DominantTopic").agg(
            F.count("*").alias("DocumentCount"),
            F.avg("Score").alias("AvgScore"),
            F.avg("ViewCount").alias("AvgViews")
        ).orderBy("DominantTopic")
        
        return topic_counts
    
    def get_topic_trends(
        self,
        df: DataFrame,
        time_column: str = "CreationDate",
        granularity: str = "month"
    ) -> DataFrame:
        """
        Analisis tren topik over time.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame dengan TopicDistribution dan waktu
        time_column : str
            Kolom waktu
        granularity : str
            Granularitas: day, week, month, year
            
        Returns
        -------
        DataFrame
            Trend topik per waktu
        """
        if granularity == "day":
            time_expr = F.to_date(time_column)
        elif granularity == "week":
            time_expr = F.date_trunc("week", time_column)
        elif granularity == "month":
            time_expr = F.date_trunc("month", time_column)
        elif granularity == "year":
            time_expr = F.date_trunc("year", time_column)
        else:
            raise ValueError(f"Unknown granularity: {granularity}")
        
        # Aggregate by time and topic
        trends = df.groupBy(
            time_expr.alias("Period"),
            "DominantTopic"
        ).agg(
            F.count("*").alias("DocumentCount")
        ).orderBy("Period", "DominantTopic")
        
        return trends
    
    def cluster_by_topic(
        self,
        df: DataFrame,
        num_clusters: int = 10,
        features_column: str = "TfidfFeatures"
    ) -> DataFrame:
        """
        Clustering dokumen menggunakan Bisecting K-Means.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame dengan fitur
        num_clusters : int
            Jumlah cluster
        features_column : str
            Kolom fitur
            
        Returns
        -------
        DataFrame
            DataFrame dengan ClusterId
        """
        logger.info(f"Clustering with {num_clusters} clusters")
        
        bkm = BisectingKMeans(
            k=num_clusters,
            featuresCol=features_column,
            predictionCol="ClusterId",
            maxIter=100
        )
        
        model = bkm.fit(df)
        df_clustered = model.transform(df)
        
        # Log cluster sizes
        cluster_sizes = df_clustered.groupBy("ClusterId").count().collect()
        for row in cluster_sizes:
            logger.info(f"Cluster {row['ClusterId']}: {row['count']} documents")
        
        return df_clustered
    
    def get_topic_coherence(self) -> float:
        """
        Menghitung topic coherence score.
        
        Returns
        -------
        float
            Coherence score (approximation)
        """
        if self.lda_model is None:
            raise ValueError("LDA model not fitted yet")
        
        # Use log perplexity as proxy for coherence
        # Lower perplexity = better coherence
        # Note: This is a simplified measure
        # For true coherence, external libraries like gensim are better
        
        return -self.lda_model.logPerplexity(
            self.cv_model.transform(
                self.spark.createDataFrame([], self.cv_model.getInputCol())
            )
        )
    
    def save_model(self, path: str) -> None:
        """
        Menyimpan model LDA.
        
        Parameters
        ----------
        path : str
            Path untuk menyimpan model
        """
        if self.lda_model is None:
            raise ValueError("No model to save")
        
        self.lda_model.save(f"{path}/lda_model")
        self.cv_model.save(f"{path}/cv_model")
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Memuat model LDA yang tersimpan.
        
        Parameters
        ----------
        path : str
            Path model
        """
        from pyspark.ml.clustering import LDAModel
        from pyspark.ml.feature import CountVectorizerModel
        
        self.lda_model = LDAModel.load(f"{path}/lda_model")
        self.cv_model = CountVectorizerModel.load(f"{path}/cv_model")
        self.vocabulary = self.cv_model.vocabulary
        logger.info(f"Model loaded from {path}")
