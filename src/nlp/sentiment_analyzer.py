"""
Sentiment Analyzer Module
=========================
Analisis sentimen untuk komentar dan teks Stack Overflow.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StringType, StructType, StructField
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analisis sentimen untuk teks Stack Overflow.
    
    Menggunakan:
    - VADER (lexicon-based)
    - TextBlob (pattern-based)
    - Custom technical sentiment
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize Sentiment Analyzer.
        
        Parameters
        ----------
        spark : SparkSession
            Active Spark session
        """
        self.spark = spark
        self._initialize_lexicon()
        self._register_udfs()
    
    def _initialize_lexicon(self):
        """Initialize sentiment lexicon untuk technical domain."""
        
        # Technical sentiment words
        self.positive_words = {
            'solved', 'works', 'working', 'fixed', 'solution', 'helpful',
            'thanks', 'thank', 'perfect', 'great', 'excellent', 'awesome',
            'correct', 'right', 'best', 'better', 'good', 'nice', 'clean',
            'elegant', 'efficient', 'fast', 'simple', 'easy', 'clear',
            'useful', 'amazing', 'brilliant', 'love', 'recommend'
        }
        
        self.negative_words = {
            'error', 'bug', 'broken', 'wrong', 'bad', 'fail', 'failed',
            'issue', 'problem', 'crash', 'crashes', 'slow', 'complex',
            'confusing', 'confused', 'difficult', 'hard', 'stuck', 'help',
            'frustrating', 'annoying', 'terrible', 'horrible', 'worst',
            'deprecated', 'outdated', 'useless', 'hate', 'ugly', 'messy'
        }
        
        # Intensifiers
        self.intensifiers = {
            'very': 1.5, 'really': 1.5, 'extremely': 2.0, 'absolutely': 2.0,
            'totally': 1.5, 'completely': 1.5, 'highly': 1.5, 'super': 1.5
        }
        
        # Negators
        self.negators = {'not', 'no', 'never', 'neither', 'nobody', 'nothing',
                         "don't", "doesn't", "didn't", "won't", "wouldn't",
                         "can't", "cannot", "couldn't", "shouldn't"}
    
    def _register_udfs(self):
        """Register UDFs untuk sentiment analysis."""
        
        positive = self.positive_words
        negative = self.negative_words
        negators = self.negators
        intensifiers = self.intensifiers
        
        @F.udf(returnType=FloatType())
        def calculate_sentiment(text):
            if text is None:
                return 0.0
            
            words = text.lower().split()
            score = 0.0
            word_count = 0
            
            prev_word = ""
            for word in words:
                multiplier = 1.0
                
                # Check for intensifier
                if prev_word in intensifiers:
                    multiplier = intensifiers.get(prev_word, 1.0)
                
                # Check for negator
                if prev_word in negators:
                    multiplier *= -1.0
                
                if word in positive:
                    score += 1.0 * multiplier
                    word_count += 1
                elif word in negative:
                    score -= 1.0 * multiplier
                    word_count += 1
                
                prev_word = word
            
            # Normalize by word count
            if word_count > 0:
                score = score / word_count
            
            # Clamp to [-1, 1]
            return max(-1.0, min(1.0, score))
        
        self._sentiment_udf = calculate_sentiment
        
        @F.udf(returnType=StringType())
        def get_sentiment_label(score):
            if score is None:
                return "neutral"
            if score > 0.1:
                return "positive"
            elif score < -0.1:
                return "negative"
            else:
                return "neutral"
        
        self._label_udf = get_sentiment_label
    
    def analyze_sentiment(
        self,
        df: DataFrame,
        text_column: str = "Text",
        output_column: str = "SentimentScore"
    ) -> DataFrame:
        """
        Menganalisis sentimen dari kolom teks.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame
        text_column : str
            Kolom teks
        output_column : str
            Kolom output score
            
        Returns
        -------
        DataFrame
            DataFrame dengan skor sentimen
        """
        logger.info(f"Analyzing sentiment for column: {text_column}")
        
        df = df.withColumn(output_column, self._sentiment_udf(F.col(text_column)))
        df = df.withColumn(f"{output_column}Label", self._label_udf(F.col(output_column)))
        
        return df
    
    def analyze_comments(self, comments_df: DataFrame) -> DataFrame:
        """
        Menganalisis sentimen komentar Stack Overflow.
        
        Parameters
        ----------
        comments_df : DataFrame
            Comments DataFrame
            
        Returns
        -------
        DataFrame
            DataFrame dengan analisis sentimen
        """
        logger.info("Analyzing comment sentiments")
        
        # Analyze text sentiment
        df = self.analyze_sentiment(
            comments_df, 
            text_column="TextClean" if "TextClean" in comments_df.columns else "Text"
        )
        
        return df
    
    def get_sentiment_summary(
        self,
        df: DataFrame,
        group_by: str = None
    ) -> DataFrame:
        """
        Mendapatkan ringkasan sentimen.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame dengan SentimentScore
        group_by : str, optional
            Kolom untuk grouping
            
        Returns
        -------
        DataFrame
            Summary statistics
        """
        agg_exprs = [
            F.count("*").alias("TotalCount"),
            F.avg("SentimentScore").alias("AvgSentiment"),
            F.stddev("SentimentScore").alias("StdSentiment"),
            F.min("SentimentScore").alias("MinSentiment"),
            F.max("SentimentScore").alias("MaxSentiment"),
            F.sum(F.when(F.col("SentimentScoreLabel") == "positive", 1).otherwise(0))
                .alias("PositiveCount"),
            F.sum(F.when(F.col("SentimentScoreLabel") == "negative", 1).otherwise(0))
                .alias("NegativeCount"),
            F.sum(F.when(F.col("SentimentScoreLabel") == "neutral", 1).otherwise(0))
                .alias("NeutralCount")
        ]
        
        if group_by:
            summary = df.groupBy(group_by).agg(*agg_exprs)
        else:
            summary = df.agg(*agg_exprs)
        
        # Calculate percentages
        summary = summary \
            .withColumn("PositivePercent", 
                F.col("PositiveCount") / F.col("TotalCount") * 100) \
            .withColumn("NegativePercent",
                F.col("NegativeCount") / F.col("TotalCount") * 100) \
            .withColumn("NeutralPercent",
                F.col("NeutralCount") / F.col("TotalCount") * 100)
        
        return summary
    
    def sentiment_by_tag(
        self,
        posts_df: DataFrame,
        comments_df: DataFrame
    ) -> DataFrame:
        """
        Analisis sentimen per tag teknologi.
        
        Parameters
        ----------
        posts_df : DataFrame
            Posts dengan TagsList
        comments_df : DataFrame
            Comments dengan sentimen
            
        Returns
        -------
        DataFrame
            Sentimen agregat per tag
        """
        logger.info("Analyzing sentiment by technology tag")
        
        # Join comments with posts to get tags
        joined = comments_df.join(
            posts_df.select("Id", "TagsList"),
            comments_df.PostId == posts_df.Id,
            "left"
        )
        
        # Explode tags
        exploded = joined.select(
            F.explode("TagsList").alias("Tag"),
            "SentimentScore"
        )
        
        # Aggregate by tag
        tag_sentiment = exploded.groupBy("Tag").agg(
            F.count("*").alias("CommentCount"),
            F.avg("SentimentScore").alias("AvgSentiment"),
            F.stddev("SentimentScore").alias("StdSentiment")
        ).orderBy(F.desc("CommentCount"))
        
        return tag_sentiment
    
    def sentiment_over_time(
        self,
        df: DataFrame,
        time_column: str = "CreationDate",
        granularity: str = "month"
    ) -> DataFrame:
        """
        Analisis tren sentimen over time.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame dengan sentimen dan waktu
        time_column : str
            Kolom waktu
        granularity : str
            Granularitas waktu
            
        Returns
        -------
        DataFrame
            Tren sentimen
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
        
        trend = df.groupBy(time_expr.alias("Period")).agg(
            F.count("*").alias("Count"),
            F.avg("SentimentScore").alias("AvgSentiment")
        ).orderBy("Period")
        
        return trend
    
    def detect_toxic_comments(
        self,
        df: DataFrame,
        threshold: float = -0.5
    ) -> DataFrame:
        """
        Mendeteksi komentar yang berpotensi toxic.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame dengan sentimen
        threshold : float
            Threshold skor untuk toxic
            
        Returns
        -------
        DataFrame
            Komentar yang berpotensi toxic
        """
        return df.filter(F.col("SentimentScore") < threshold) \
            .orderBy(F.asc("SentimentScore"))
    
    def analyze_with_vader(
        self,
        df: DataFrame,
        text_column: str = "Text"
    ) -> DataFrame:
        """
        Analisis sentimen menggunakan VADER.
        
        Note: Memerlukan nltk.sentiment.vader
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame
        text_column : str
            Kolom teks
            
        Returns
        -------
        DataFrame
            DataFrame dengan VADER scores
        """
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            
            # Broadcast VADER analyzer
            sia = SentimentIntensityAnalyzer()
            sia_broadcast = self.spark.sparkContext.broadcast(sia)
            
            schema = StructType([
                StructField("compound", FloatType(), True),
                StructField("positive", FloatType(), True),
                StructField("negative", FloatType(), True),
                StructField("neutral", FloatType(), True)
            ])
            
            @F.udf(returnType=schema)
            def vader_sentiment(text):
                if text is None:
                    return (0.0, 0.0, 0.0, 0.0)
                
                scores = sia_broadcast.value.polarity_scores(text)
                return (
                    float(scores['compound']),
                    float(scores['pos']),
                    float(scores['neg']),
                    float(scores['neu'])
                )
            
            df = df.withColumn("VaderScores", vader_sentiment(F.col(text_column)))
            df = df.withColumn("VaderCompound", F.col("VaderScores.compound"))
            df = df.withColumn("VaderPositive", F.col("VaderScores.positive"))
            df = df.withColumn("VaderNegative", F.col("VaderScores.negative"))
            df = df.withColumn("VaderNeutral", F.col("VaderScores.neutral"))
            df = df.drop("VaderScores")
            
            return df
            
        except ImportError:
            logger.warning("VADER not available. Install nltk and download vader_lexicon.")
            return self.analyze_sentiment(df, text_column)
