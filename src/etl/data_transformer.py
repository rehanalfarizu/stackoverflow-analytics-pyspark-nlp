"""
Data Transformer Module
=======================
Transformasi dan pembersihan data Stack Overflow.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, IntegerType
from typing import List, Optional
import re
import logging

logger = logging.getLogger(__name__)


class DataTransformer:
    """
    Transformer untuk membersihkan dan mentransformasi data Stack Overflow.
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize Data Transformer.
        
        Parameters
        ----------
        spark : SparkSession
            Active Spark session
        """
        self.spark = spark
        self._register_udfs()
    
    def _register_udfs(self):
        """Register User Defined Functions."""
        
        # UDF untuk extract tags dari format <tag1><tag2><tag3>
        @F.udf(returnType=ArrayType(StringType()))
        def extract_tags(tags_str):
            if tags_str is None:
                return []
            # Remove < dan > lalu split
            tags = re.findall(r'<([^>]+)>', tags_str)
            return tags
        
        self._extract_tags_udf = extract_tags
        
        # UDF untuk menghitung panjang kata
        @F.udf(returnType=IntegerType())
        def word_count(text):
            if text is None:
                return 0
            return len(text.split())
        
        self._word_count_udf = word_count
        
        # UDF untuk remove HTML tags
        @F.udf(returnType=StringType())
        def remove_html(text):
            if text is None:
                return ""
            clean = re.sub(r'<[^>]+>', ' ', text)
            clean = re.sub(r'\s+', ' ', clean)
            return clean.strip()
        
        self._remove_html_udf = remove_html
        
        # UDF untuk extract code blocks
        @F.udf(returnType=StringType())
        def extract_code(text):
            if text is None:
                return ""
            code_blocks = re.findall(r'<code>(.*?)</code>', text, re.DOTALL)
            return '\n'.join(code_blocks)
        
        self._extract_code_udf = extract_code
    
    def clean_posts(self, df: DataFrame) -> DataFrame:
        """
        Membersihkan dan mentransformasi data Posts.
        
        Parameters
        ----------
        df : DataFrame
            Raw posts DataFrame
            
        Returns
        -------
        DataFrame
            Cleaned posts DataFrame dengan kolom tambahan
        """
        logger.info("Cleaning posts data...")
        
        cleaned = df \
            .withColumn("CreationDate", F.to_timestamp("CreationDate")) \
            .withColumn("LastActivityDate", F.to_timestamp("LastActivityDate")) \
            .withColumn("LastEditDate", F.to_timestamp("LastEditDate")) \
            .withColumn("ClosedDate", F.to_timestamp("ClosedDate")) \
            .withColumn("Year", F.year("CreationDate")) \
            .withColumn("Month", F.month("CreationDate")) \
            .withColumn("DayOfWeek", F.dayofweek("CreationDate")) \
            .withColumn("Hour", F.hour("CreationDate")) \
            .withColumn("TagsList", self._extract_tags_udf("Tags")) \
            .withColumn("NumTags", F.size("TagsList")) \
            .withColumn("BodyClean", self._remove_html_udf("Body")) \
            .withColumn("BodyLength", F.length("BodyClean")) \
            .withColumn("BodyWordCount", self._word_count_udf("BodyClean")) \
            .withColumn("TitleLength", F.length("Title")) \
            .withColumn("TitleWordCount", self._word_count_udf("Title")) \
            .withColumn("CodeBlocks", self._extract_code_udf("Body")) \
            .withColumn("HasCode", F.when(F.length("CodeBlocks") > 0, 1).otherwise(0)) \
            .withColumn("HasAcceptedAnswer", 
                       F.when(F.col("AcceptedAnswerId").isNotNull(), 1).otherwise(0)) \
            .withColumn("IsQuestion", F.when(F.col("PostTypeId") == 1, 1).otherwise(0)) \
            .withColumn("IsAnswer", F.when(F.col("PostTypeId") == 2, 1).otherwise(0))
        
        # Fill nulls
        cleaned = cleaned \
            .fillna({'Score': 0, 'ViewCount': 0, 'AnswerCount': 0, 
                    'CommentCount': 0, 'FavoriteCount': 0})
        
        logger.info(f"Cleaned {cleaned.count()} posts")
        return cleaned
    
    def clean_users(self, df: DataFrame) -> DataFrame:
        """
        Membersihkan data Users.
        
        Parameters
        ----------
        df : DataFrame
            Raw users DataFrame
            
        Returns
        -------
        DataFrame
            Cleaned users DataFrame
        """
        logger.info("Cleaning users data...")
        
        cleaned = df \
            .withColumn("CreationDate", F.to_timestamp("CreationDate")) \
            .withColumn("LastAccessDate", F.to_timestamp("LastAccessDate")) \
            .withColumn("AccountAge", 
                       F.datediff(F.current_date(), "CreationDate")) \
            .withColumn("AboutMeClean", self._remove_html_udf("AboutMe")) \
            .withColumn("AboutMeLength", F.length("AboutMeClean")) \
            .withColumn("HasWebsite", 
                       F.when(F.col("WebsiteUrl").isNotNull(), 1).otherwise(0)) \
            .withColumn("HasLocation", 
                       F.when(F.col("Location").isNotNull(), 1).otherwise(0)) \
            .fillna({'Reputation': 0, 'Views': 0, 'UpVotes': 0, 'DownVotes': 0})
        
        return cleaned
    
    def clean_comments(self, df: DataFrame) -> DataFrame:
        """
        Membersihkan data Comments.
        
        Parameters
        ----------
        df : DataFrame
            Raw comments DataFrame
            
        Returns
        -------
        DataFrame
            Cleaned comments DataFrame
        """
        logger.info("Cleaning comments data...")
        
        cleaned = df \
            .withColumn("CreationDate", F.to_timestamp("CreationDate")) \
            .withColumn("TextClean", self._remove_html_udf("Text")) \
            .withColumn("TextLength", F.length("TextClean")) \
            .withColumn("TextWordCount", self._word_count_udf("TextClean")) \
            .fillna({'Score': 0})
        
        return cleaned
    
    def filter_questions(self, posts_df: DataFrame) -> DataFrame:
        """Filter hanya pertanyaan (PostTypeId = 1)."""
        return posts_df.filter(F.col("PostTypeId") == 1)
    
    def filter_answers(self, posts_df: DataFrame) -> DataFrame:
        """Filter hanya jawaban (PostTypeId = 2)."""
        return posts_df.filter(F.col("PostTypeId") == 2)
    
    def filter_by_date_range(
        self, 
        df: DataFrame, 
        start_date: str, 
        end_date: str,
        date_column: str = "CreationDate"
    ) -> DataFrame:
        """
        Filter data berdasarkan rentang tanggal.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame
        start_date : str
            Tanggal mulai (format: YYYY-MM-DD)
        end_date : str
            Tanggal akhir (format: YYYY-MM-DD)
        date_column : str
            Nama kolom tanggal
            
        Returns
        -------
        DataFrame
            Filtered DataFrame
        """
        return df.filter(
            (F.col(date_column) >= start_date) & 
            (F.col(date_column) <= end_date)
        )
    
    def filter_by_tags(
        self, 
        posts_df: DataFrame, 
        tags: List[str],
        match_all: bool = False
    ) -> DataFrame:
        """
        Filter posts berdasarkan tags.
        
        Parameters
        ----------
        posts_df : DataFrame
            Posts DataFrame dengan kolom TagsList
        tags : List[str]
            List of tags untuk filter
        match_all : bool
            Jika True, harus match semua tags. Jika False, match salah satu.
            
        Returns
        -------
        DataFrame
            Filtered DataFrame
        """
        if match_all:
            # Harus memiliki semua tags
            for tag in tags:
                posts_df = posts_df.filter(F.array_contains("TagsList", tag))
        else:
            # Cukup salah satu tag
            condition = F.array_contains("TagsList", tags[0])
            for tag in tags[1:]:
                condition = condition | F.array_contains("TagsList", tag)
            posts_df = posts_df.filter(condition)
        
        return posts_df
    
    def aggregate_by_tag(self, posts_df: DataFrame) -> DataFrame:
        """
        Agregasi statistik per tag.
        
        Parameters
        ----------
        posts_df : DataFrame
            Posts DataFrame dengan kolom TagsList
            
        Returns
        -------
        DataFrame
            Agregasi per tag dengan statistik
        """
        # Explode tags
        exploded = posts_df.select(
            F.explode("TagsList").alias("Tag"),
            "Id", "Score", "ViewCount", "AnswerCount", 
            "CommentCount", "HasAcceptedAnswer", "Year", "Month"
        )
        
        # Aggregate
        tag_stats = exploded.groupBy("Tag").agg(
            F.count("Id").alias("QuestionCount"),
            F.avg("Score").alias("AvgScore"),
            F.sum("Score").alias("TotalScore"),
            F.avg("ViewCount").alias("AvgViews"),
            F.sum("ViewCount").alias("TotalViews"),
            F.avg("AnswerCount").alias("AvgAnswers"),
            F.avg("HasAcceptedAnswer").alias("AcceptedAnswerRate"),
            F.min("Year").alias("FirstYear"),
            F.max("Year").alias("LastYear")
        ).orderBy(F.desc("QuestionCount"))
        
        return tag_stats
    
    def aggregate_by_time(
        self, 
        df: DataFrame, 
        time_column: str = "CreationDate",
        granularity: str = "month"
    ) -> DataFrame:
        """
        Agregasi data berdasarkan waktu.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame
        time_column : str
            Nama kolom waktu
        granularity : str
            Granularitas: 'day', 'week', 'month', 'year'
            
        Returns
        -------
        DataFrame
            Time series aggregation
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
        
        result = df.groupBy(time_expr.alias("Period")).agg(
            F.count("*").alias("Count"),
            F.avg("Score").alias("AvgScore"),
            F.sum("Score").alias("TotalScore")
        ).orderBy("Period")
        
        return result
    
    def create_quality_labels(
        self, 
        posts_df: DataFrame,
        score_threshold: int = 5,
        view_threshold: int = 1000
    ) -> DataFrame:
        """
        Membuat label kualitas untuk supervised learning.
        
        Parameters
        ----------
        posts_df : DataFrame
            Posts DataFrame
        score_threshold : int
            Threshold skor untuk high quality
        view_threshold : int
            Threshold views untuk high quality
            
        Returns
        -------
        DataFrame
            DataFrame dengan kolom QualityLabel
        """
        return posts_df.withColumn(
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
    
    def deduplicate_posts(
        self, 
        posts_df: DataFrame,
        columns: List[str] = None
    ) -> DataFrame:
        """
        Menghapus post duplikat.
        
        Parameters
        ----------
        posts_df : DataFrame
            Posts DataFrame
        columns : List[str], optional
            Kolom untuk cek duplikasi
            
        Returns
        -------
        DataFrame
            Deduplicated DataFrame
        """
        if columns is None:
            columns = ["Title", "BodyClean"]
        
        return posts_df.dropDuplicates(columns)
