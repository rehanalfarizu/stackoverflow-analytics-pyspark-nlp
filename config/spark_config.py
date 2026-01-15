"""
Spark Configuration Module
===========================
Konfigurasi Apache Spark untuk Stack Overflow Analytics Pipeline.
"""

from pyspark.sql import SparkSession
from pyspark import SparkConf
import os


class SparkConfig:
    """Konfigurasi Spark untuk berbagai environment."""
    
    # Default configurations
    DEFAULT_APP_NAME = "StackOverflowAnalytics"
    DEFAULT_MASTER = "local[*]"
    
    # Memory configurations
    DRIVER_MEMORY = "4g"
    EXECUTOR_MEMORY = "8g"
    MAX_RESULT_SIZE = "2g"
    
    # Parallelism
    DEFAULT_PARALLELISM = 200
    SQL_SHUFFLE_PARTITIONS = 200
    
    # Serialization
    SERIALIZER = "org.apache.spark.serializer.KryoSerializer"
    
    @classmethod
    def get_local_config(cls) -> SparkConf:
        """Konfigurasi untuk local development."""
        conf = SparkConf()
        conf.setAppName(cls.DEFAULT_APP_NAME)
        conf.setMaster(cls.DEFAULT_MASTER)
        
        # Memory settings
        conf.set("spark.driver.memory", cls.DRIVER_MEMORY)
        conf.set("spark.executor.memory", cls.EXECUTOR_MEMORY)
        conf.set("spark.driver.maxResultSize", cls.MAX_RESULT_SIZE)
        
        # Performance tuning
        conf.set("spark.sql.shuffle.partitions", str(cls.SQL_SHUFFLE_PARTITIONS))
        conf.set("spark.default.parallelism", str(cls.DEFAULT_PARALLELISM))
        
        # Serialization
        conf.set("spark.serializer", cls.SERIALIZER)
        
        # Adaptive query execution
        conf.set("spark.sql.adaptive.enabled", "true")
        conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        
        # XML parsing support
        conf.set("spark.jars.packages", "com.databricks:spark-xml_2.12:0.17.0")
        
        return conf
    
    @classmethod
    def get_cluster_config(cls, master_url: str, num_executors: int = 4) -> SparkConf:
        """Konfigurasi untuk cluster deployment."""
        conf = SparkConf()
        conf.setAppName(cls.DEFAULT_APP_NAME)
        conf.setMaster(master_url)
        
        # Cluster-specific settings
        conf.set("spark.executor.instances", str(num_executors))
        conf.set("spark.executor.memory", "16g")
        conf.set("spark.executor.cores", "4")
        conf.set("spark.driver.memory", "8g")
        
        # Dynamic allocation
        conf.set("spark.dynamicAllocation.enabled", "true")
        conf.set("spark.dynamicAllocation.minExecutors", "2")
        conf.set("spark.dynamicAllocation.maxExecutors", str(num_executors * 2))
        
        # Shuffle optimization
        conf.set("spark.sql.shuffle.partitions", "400")
        
        # Serialization
        conf.set("spark.serializer", cls.SERIALIZER)
        conf.set("spark.kryoserializer.buffer.max", "1024m")
        
        return conf


def create_spark_session(
    app_name: str = None,
    master: str = None,
    config: SparkConf = None
) -> SparkSession:
    """
    Membuat SparkSession dengan konfigurasi yang sesuai.
    
    Parameters
    ----------
    app_name : str, optional
        Nama aplikasi Spark
    master : str, optional
        URL master Spark (local[*], spark://host:port, yarn)
    config : SparkConf, optional
        Custom SparkConf object
        
    Returns
    -------
    SparkSession
        Configured Spark session
    """
    if config is None:
        config = SparkConfig.get_local_config()
    
    if app_name:
        config.setAppName(app_name)
    
    if master:
        config.setMaster(master)
    
    # Build session
    builder = SparkSession.builder.config(conf=config)
    
    # Enable Hive support if available
    try:
        spark = builder.enableHiveSupport().getOrCreate()
    except Exception:
        spark = builder.getOrCreate()
    
    # Set log level
    spark.sparkContext.setLogLevel("WARN")
    
    return spark


def get_spark_session() -> SparkSession:
    """
    Mendapatkan atau membuat SparkSession singleton.
    
    Returns
    -------
    SparkSession
        Active Spark session
    """
    return SparkSession.builder.getOrCreate()


# Schema definitions for Stack Overflow data
POSTS_SCHEMA = """
    Id INT,
    PostTypeId INT,
    AcceptedAnswerId INT,
    ParentId INT,
    CreationDate TIMESTAMP,
    DeletionDate TIMESTAMP,
    Score INT,
    ViewCount INT,
    Body STRING,
    OwnerUserId INT,
    OwnerDisplayName STRING,
    LastEditorUserId INT,
    LastEditorDisplayName STRING,
    LastEditDate TIMESTAMP,
    LastActivityDate TIMESTAMP,
    Title STRING,
    Tags STRING,
    AnswerCount INT,
    CommentCount INT,
    FavoriteCount INT,
    ClosedDate TIMESTAMP,
    CommunityOwnedDate TIMESTAMP,
    ContentLicense STRING
"""

USERS_SCHEMA = """
    Id INT,
    Reputation INT,
    CreationDate TIMESTAMP,
    DisplayName STRING,
    LastAccessDate TIMESTAMP,
    WebsiteUrl STRING,
    Location STRING,
    AboutMe STRING,
    Views INT,
    UpVotes INT,
    DownVotes INT,
    ProfileImageUrl STRING,
    AccountId INT
"""

COMMENTS_SCHEMA = """
    Id INT,
    PostId INT,
    Score INT,
    Text STRING,
    CreationDate TIMESTAMP,
    UserDisplayName STRING,
    UserId INT,
    ContentLicense STRING
"""

TAGS_SCHEMA = """
    Id INT,
    TagName STRING,
    Count INT,
    ExcerptPostId INT,
    WikiPostId INT
"""

VOTES_SCHEMA = """
    Id INT,
    PostId INT,
    VoteTypeId INT,
    UserId INT,
    CreationDate TIMESTAMP,
    BountyAmount INT
"""

BADGES_SCHEMA = """
    Id INT,
    UserId INT,
    Name STRING,
    Date TIMESTAMP,
    Class INT,
    TagBased BOOLEAN
"""
