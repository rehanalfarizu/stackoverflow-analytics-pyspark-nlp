"""
Data Loader Module
==================
Module untuk menyimpan dan memuat data dalam berbagai format.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from typing import List, Optional
import os
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loader untuk menyimpan dan memuat data dalam berbagai format.
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize Data Loader.
        
        Parameters
        ----------
        spark : SparkSession
            Active Spark session
        """
        self.spark = spark
    
    def save_parquet(
        self,
        df: DataFrame,
        path: str,
        partition_by: List[str] = None,
        mode: str = "overwrite",
        compression: str = "snappy"
    ) -> None:
        """
        Menyimpan DataFrame ke format Parquet.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame untuk disimpan
        path : str
            Path output
        partition_by : List[str], optional
            Kolom untuk partisi
        mode : str
            Save mode: overwrite, append, ignore, error
        compression : str
            Kompresi: snappy, gzip, lzo, none
        """
        logger.info(f"Saving DataFrame to Parquet: {path}")
        
        writer = df.write.mode(mode).option("compression", compression)
        
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        
        writer.parquet(path)
        logger.info(f"Saved to {path}")
    
    def load_parquet(
        self,
        path: str,
        columns: List[str] = None
    ) -> DataFrame:
        """
        Memuat data dari file Parquet.
        
        Parameters
        ----------
        path : str
            Path ke file Parquet
        columns : List[str], optional
            Kolom yang akan dimuat
            
        Returns
        -------
        DataFrame
            Loaded DataFrame
        """
        logger.info(f"Loading Parquet from: {path}")
        
        df = self.spark.read.parquet(path)
        
        if columns:
            df = df.select(*columns)
        
        return df
    
    def save_csv(
        self,
        df: DataFrame,
        path: str,
        header: bool = True,
        mode: str = "overwrite"
    ) -> None:
        """
        Menyimpan DataFrame ke format CSV.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame untuk disimpan
        path : str
            Path output
        header : bool
            Include header
        mode : str
            Save mode
        """
        logger.info(f"Saving DataFrame to CSV: {path}")
        
        df.write.mode(mode) \
            .option("header", str(header).lower()) \
            .csv(path)
        
        logger.info(f"Saved to {path}")
    
    def load_csv(
        self,
        path: str,
        header: bool = True,
        infer_schema: bool = True
    ) -> DataFrame:
        """
        Memuat data dari file CSV.
        
        Parameters
        ----------
        path : str
            Path ke file CSV
        header : bool
            File memiliki header
        infer_schema : bool
            Inferensi schema otomatis
            
        Returns
        -------
        DataFrame
            Loaded DataFrame
        """
        logger.info(f"Loading CSV from: {path}")
        
        return self.spark.read \
            .option("header", str(header).lower()) \
            .option("inferSchema", str(infer_schema).lower()) \
            .csv(path)
    
    def save_json(
        self,
        df: DataFrame,
        path: str,
        mode: str = "overwrite"
    ) -> None:
        """
        Menyimpan DataFrame ke format JSON.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame untuk disimpan
        path : str
            Path output
        mode : str
            Save mode
        """
        logger.info(f"Saving DataFrame to JSON: {path}")
        df.write.mode(mode).json(path)
        logger.info(f"Saved to {path}")
    
    def load_json(self, path: str) -> DataFrame:
        """
        Memuat data dari file JSON.
        
        Parameters
        ----------
        path : str
            Path ke file JSON
            
        Returns
        -------
        DataFrame
            Loaded DataFrame
        """
        logger.info(f"Loading JSON from: {path}")
        return self.spark.read.json(path)
    
    def cache_dataframe(self, df: DataFrame, name: str) -> DataFrame:
        """
        Cache DataFrame dan register sebagai temp view.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame untuk di-cache
        name : str
            Nama untuk temp view
            
        Returns
        -------
        DataFrame
            Cached DataFrame
        """
        df.cache()
        df.createOrReplaceTempView(name)
        logger.info(f"Cached and registered temp view: {name}")
        return df
    
    def get_data_info(self, df: DataFrame) -> dict:
        """
        Mendapatkan informasi tentang DataFrame.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame
            
        Returns
        -------
        dict
            Informasi DataFrame
        """
        return {
            'num_rows': df.count(),
            'num_columns': len(df.columns),
            'columns': df.columns,
            'schema': df.schema.simpleString(),
            'partitions': df.rdd.getNumPartitions()
        }
    
    def repartition_and_save(
        self,
        df: DataFrame,
        path: str,
        num_partitions: int = None,
        partition_by: List[str] = None
    ) -> None:
        """
        Repartisi DataFrame dan simpan.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame
        path : str
            Path output
        num_partitions : int, optional
            Jumlah partisi
        partition_by : List[str], optional
            Kolom untuk partisi
        """
        if num_partitions:
            df = df.repartition(num_partitions)
        
        self.save_parquet(df, path, partition_by=partition_by)
    
    def sample_data(
        self,
        df: DataFrame,
        fraction: float = 0.1,
        seed: int = 42
    ) -> DataFrame:
        """
        Mengambil sample dari DataFrame.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame
        fraction : float
            Fraksi data (0.0 - 1.0)
        seed : int
            Random seed
            
        Returns
        -------
        DataFrame
            Sampled DataFrame
        """
        return df.sample(withReplacement=False, fraction=fraction, seed=seed)
    
    def create_sample_dataset(
        self,
        source_path: str,
        output_path: str,
        fraction: float = 0.01
    ) -> None:
        """
        Membuat sample dataset dari data penuh.
        
        Parameters
        ----------
        source_path : str
            Path ke data penuh
        output_path : str
            Path output sample
        fraction : float
            Fraksi sample
        """
        logger.info(f"Creating sample dataset with fraction {fraction}")
        
        df = self.load_parquet(source_path)
        sample_df = self.sample_data(df, fraction=fraction)
        self.save_parquet(sample_df, output_path)
        
        logger.info(f"Sample dataset created at {output_path}")
