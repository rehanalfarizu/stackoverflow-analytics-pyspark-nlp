"""
XML Parser Module
=================
Parser untuk membaca file XML Stack Overflow Data Dump.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField, IntegerType, StringType, 
    TimestampType, LongType, BooleanType
)
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class XMLParser:
    """
    Parser untuk file XML Stack Overflow.
    
    Stack Overflow Data Dump menggunakan format XML dengan atribut
    pada elemen row. Parser ini menggunakan spark-xml untuk membaca
    file XML besar secara efisien.
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize XML Parser.
        
        Parameters
        ----------
        spark : SparkSession
            Active Spark session dengan spark-xml package
        """
        self.spark = spark
        self._schemas = self._define_schemas()
    
    def _define_schemas(self) -> dict:
        """Define schemas untuk setiap tipe file XML."""
        
        posts_schema = StructType([
            StructField("_Id", IntegerType(), True),
            StructField("_PostTypeId", IntegerType(), True),
            StructField("_AcceptedAnswerId", IntegerType(), True),
            StructField("_ParentId", IntegerType(), True),
            StructField("_CreationDate", StringType(), True),
            StructField("_DeletionDate", StringType(), True),
            StructField("_Score", IntegerType(), True),
            StructField("_ViewCount", IntegerType(), True),
            StructField("_Body", StringType(), True),
            StructField("_OwnerUserId", IntegerType(), True),
            StructField("_OwnerDisplayName", StringType(), True),
            StructField("_LastEditorUserId", IntegerType(), True),
            StructField("_LastEditorDisplayName", StringType(), True),
            StructField("_LastEditDate", StringType(), True),
            StructField("_LastActivityDate", StringType(), True),
            StructField("_Title", StringType(), True),
            StructField("_Tags", StringType(), True),
            StructField("_AnswerCount", IntegerType(), True),
            StructField("_CommentCount", IntegerType(), True),
            StructField("_FavoriteCount", IntegerType(), True),
            StructField("_ClosedDate", StringType(), True),
            StructField("_CommunityOwnedDate", StringType(), True),
            StructField("_ContentLicense", StringType(), True)
        ])
        
        users_schema = StructType([
            StructField("_Id", IntegerType(), True),
            StructField("_Reputation", IntegerType(), True),
            StructField("_CreationDate", StringType(), True),
            StructField("_DisplayName", StringType(), True),
            StructField("_LastAccessDate", StringType(), True),
            StructField("_WebsiteUrl", StringType(), True),
            StructField("_Location", StringType(), True),
            StructField("_AboutMe", StringType(), True),
            StructField("_Views", IntegerType(), True),
            StructField("_UpVotes", IntegerType(), True),
            StructField("_DownVotes", IntegerType(), True),
            StructField("_ProfileImageUrl", StringType(), True),
            StructField("_AccountId", IntegerType(), True)
        ])
        
        comments_schema = StructType([
            StructField("_Id", IntegerType(), True),
            StructField("_PostId", IntegerType(), True),
            StructField("_Score", IntegerType(), True),
            StructField("_Text", StringType(), True),
            StructField("_CreationDate", StringType(), True),
            StructField("_UserDisplayName", StringType(), True),
            StructField("_UserId", IntegerType(), True),
            StructField("_ContentLicense", StringType(), True)
        ])
        
        tags_schema = StructType([
            StructField("_Id", IntegerType(), True),
            StructField("_TagName", StringType(), True),
            StructField("_Count", IntegerType(), True),
            StructField("_ExcerptPostId", IntegerType(), True),
            StructField("_WikiPostId", IntegerType(), True)
        ])
        
        votes_schema = StructType([
            StructField("_Id", IntegerType(), True),
            StructField("_PostId", IntegerType(), True),
            StructField("_VoteTypeId", IntegerType(), True),
            StructField("_UserId", IntegerType(), True),
            StructField("_CreationDate", StringType(), True),
            StructField("_BountyAmount", IntegerType(), True)
        ])
        
        badges_schema = StructType([
            StructField("_Id", IntegerType(), True),
            StructField("_UserId", IntegerType(), True),
            StructField("_Name", StringType(), True),
            StructField("_Date", StringType(), True),
            StructField("_Class", IntegerType(), True),
            StructField("_TagBased", StringType(), True)
        ])
        
        return {
            'posts': posts_schema,
            'users': users_schema,
            'comments': comments_schema,
            'tags': tags_schema,
            'votes': votes_schema,
            'badges': badges_schema
        }
    
    def parse_xml(
        self, 
        file_path: str, 
        row_tag: str = "row",
        schema: Optional[StructType] = None
    ) -> DataFrame:
        """
        Parse file XML generik.
        
        Parameters
        ----------
        file_path : str
            Path ke file XML
        row_tag : str
            Tag XML untuk setiap row (default: "row")
        schema : StructType, optional
            Schema untuk DataFrame
            
        Returns
        -------
        DataFrame
            Parsed data sebagai Spark DataFrame
        """
        logger.info(f"Parsing XML file: {file_path}")
        
        reader = self.spark.read.format("xml") \
            .option("rowTag", row_tag) \
            .option("attributePrefix", "_") \
            .option("valueTag", "_value")
        
        if schema:
            reader = reader.schema(schema)
        
        df = reader.load(file_path)
        
        # Rename columns to remove underscore prefix
        for col_name in df.columns:
            if col_name.startswith("_"):
                df = df.withColumnRenamed(col_name, col_name[1:])
        
        logger.info(f"Parsed {df.count()} rows from {file_path}")
        return df
    
    def parse_posts(self, file_path: str) -> DataFrame:
        """
        Parse Posts.xml.
        
        Parameters
        ----------
        file_path : str
            Path ke Posts.xml
            
        Returns
        -------
        DataFrame
            Posts data dengan columns:
            - Id, PostTypeId, AcceptedAnswerId, ParentId
            - CreationDate, Score, ViewCount, Body
            - OwnerUserId, Title, Tags, AnswerCount, etc.
        """
        return self.parse_xml(file_path, schema=self._schemas['posts'])
    
    def parse_users(self, file_path: str) -> DataFrame:
        """Parse Users.xml."""
        return self.parse_xml(file_path, schema=self._schemas['users'])
    
    def parse_comments(self, file_path: str) -> DataFrame:
        """Parse Comments.xml."""
        return self.parse_xml(file_path, schema=self._schemas['comments'])
    
    def parse_tags(self, file_path: str) -> DataFrame:
        """Parse Tags.xml."""
        return self.parse_xml(file_path, schema=self._schemas['tags'])
    
    def parse_votes(self, file_path: str) -> DataFrame:
        """Parse Votes.xml."""
        return self.parse_xml(file_path, schema=self._schemas['votes'])
    
    def parse_badges(self, file_path: str) -> DataFrame:
        """Parse Badges.xml."""
        return self.parse_xml(file_path, schema=self._schemas['badges'])
    
    def parse_all(self, data_dir: str) -> dict:
        """
        Parse semua file XML dalam direktori.
        
        Parameters
        ----------
        data_dir : str
            Direktori yang berisi file XML
            
        Returns
        -------
        dict
            Dictionary dengan key nama file dan value DataFrame
        """
        import os
        
        results = {}
        
        file_parsers = {
            'Posts.xml': self.parse_posts,
            'Users.xml': self.parse_users,
            'Comments.xml': self.parse_comments,
            'Tags.xml': self.parse_tags,
            'Votes.xml': self.parse_votes,
            'Badges.xml': self.parse_badges
        }
        
        for filename, parser in file_parsers.items():
            file_path = os.path.join(data_dir, filename)
            if os.path.exists(file_path):
                logger.info(f"Parsing {filename}...")
                results[filename.replace('.xml', '').lower()] = parser(file_path)
            else:
                logger.warning(f"File not found: {file_path}")
        
        return results
