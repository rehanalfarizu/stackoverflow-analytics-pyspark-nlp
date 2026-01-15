"""
Trend Forecaster Module
=======================
Forecasting tren teknologi dari data Stack Overflow.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TrendForecaster:
    """
    Forecaster untuk tren teknologi di Stack Overflow.
    
    Menganalisis dan memprediksi:
    - Teknologi yang trending
    - Teknologi yang declining
    - Seasonality patterns
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize Trend Forecaster.
        
        Parameters
        ----------
        spark : SparkSession
            Active Spark session
        """
        self.spark = spark
    
    def calculate_tag_trends(
        self,
        posts_df: DataFrame,
        time_column: str = "CreationDate",
        granularity: str = "month"
    ) -> DataFrame:
        """
        Menghitung tren per tag over time.
        
        Parameters
        ----------
        posts_df : DataFrame
            Posts dengan TagsList
        time_column : str
            Kolom waktu
        granularity : str
            Granularitas: day, week, month, year
            
        Returns
        -------
        DataFrame
            Tren per tag
        """
        logger.info("Calculating tag trends")
        
        # Time truncation
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
        
        # Explode tags and aggregate
        exploded = posts_df.select(
            F.explode("TagsList").alias("Tag"),
            time_expr.alias("Period"),
            "Score", "ViewCount"
        )
        
        trends = exploded.groupBy("Tag", "Period").agg(
            F.count("*").alias("QuestionCount"),
            F.avg("Score").alias("AvgScore"),
            F.sum("ViewCount").alias("TotalViews")
        ).orderBy("Tag", "Period")
        
        return trends
    
    def detect_trending_tags(
        self,
        trends_df: DataFrame,
        min_questions: int = 100,
        lookback_periods: int = 6
    ) -> DataFrame:
        """
        Mendeteksi tag yang trending.
        
        Parameters
        ----------
        trends_df : DataFrame
            DataFrame dari calculate_tag_trends
        min_questions : int
            Minimum pertanyaan total
        lookback_periods : int
            Periode untuk perhitungan growth
            
        Returns
        -------
        DataFrame
            Tag trending dengan growth rate
        """
        logger.info("Detecting trending tags")
        
        # Window untuk lag calculation
        window_spec = Window.partitionBy("Tag").orderBy("Period")
        
        # Calculate lagged values
        trends_with_lag = trends_df \
            .withColumn(
                "PrevCount", 
                F.lag("QuestionCount", lookback_periods).over(window_spec)
            ) \
            .withColumn(
                "Growth",
                (F.col("QuestionCount") - F.col("PrevCount")) / 
                F.greatest(F.col("PrevCount"), F.lit(1)) * 100
            )
        
        # Filter tags with enough data
        tag_totals = trends_df.groupBy("Tag").agg(
            F.sum("QuestionCount").alias("TotalQuestions")
        ).filter(F.col("TotalQuestions") >= min_questions)
        
        # Get latest period growth
        latest_period = trends_df.agg(F.max("Period")).collect()[0][0]
        
        trending = trends_with_lag \
            .filter(F.col("Period") == latest_period) \
            .join(tag_totals, "Tag") \
            .select("Tag", "QuestionCount", "Growth", "TotalQuestions") \
            .orderBy(F.desc("Growth"))
        
        return trending
    
    def detect_declining_tags(
        self,
        trends_df: DataFrame,
        min_questions: int = 100,
        lookback_periods: int = 6
    ) -> DataFrame:
        """
        Mendeteksi tag yang declining.
        
        Parameters
        ----------
        trends_df : DataFrame
            DataFrame dari calculate_tag_trends
        min_questions : int
            Minimum pertanyaan total
        lookback_periods : int
            Periode untuk perhitungan decline
            
        Returns
        -------
        DataFrame
            Tag declining dengan decline rate
        """
        logger.info("Detecting declining tags")
        
        trending = self.detect_trending_tags(
            trends_df, 
            min_questions, 
            lookback_periods
        )
        
        # Filter negative growth
        declining = trending \
            .filter(F.col("Growth") < 0) \
            .orderBy(F.asc("Growth"))
        
        return declining
    
    def calculate_moving_average(
        self,
        trends_df: DataFrame,
        window_size: int = 3
    ) -> DataFrame:
        """
        Menghitung moving average untuk smoothing.
        
        Parameters
        ----------
        trends_df : DataFrame
            Trend data
        window_size : int
            Ukuran window
            
        Returns
        -------
        DataFrame
            DataFrame dengan moving average
        """
        window_spec = Window.partitionBy("Tag") \
            .orderBy("Period") \
            .rowsBetween(-window_size + 1, 0)
        
        return trends_df.withColumn(
            "QuestionCountMA",
            F.avg("QuestionCount").over(window_spec)
        )
    
    def detect_seasonality(
        self,
        trends_df: DataFrame
    ) -> DataFrame:
        """
        Mendeteksi pola seasonality.
        
        Parameters
        ----------
        trends_df : DataFrame
            Trend data dengan Period
            
        Returns
        -------
        DataFrame
            Pola seasonal per bulan
        """
        logger.info("Detecting seasonality patterns")
        
        # Extract month
        seasonal = trends_df \
            .withColumn("Month", F.month("Period")) \
            .groupBy("Tag", "Month") \
            .agg(
                F.avg("QuestionCount").alias("AvgQuestions"),
                F.stddev("QuestionCount").alias("StdQuestions")
            ) \
            .orderBy("Tag", "Month")
        
        # Calculate seasonal index
        tag_avg = trends_df.groupBy("Tag").agg(
            F.avg("QuestionCount").alias("OverallAvg")
        )
        
        seasonal = seasonal.join(tag_avg, "Tag") \
            .withColumn(
                "SeasonalIndex",
                F.col("AvgQuestions") / F.col("OverallAvg")
            )
        
        return seasonal
    
    def forecast_simple(
        self,
        trends_df: DataFrame,
        tag: str,
        periods: int = 6
    ) -> DataFrame:
        """
        Simple forecasting menggunakan exponential smoothing.
        
        Parameters
        ----------
        trends_df : DataFrame
            Trend data
        tag : str
            Tag untuk forecast
        periods : int
            Jumlah periode forecast
            
        Returns
        -------
        DataFrame
            Forecast values
        """
        logger.info(f"Forecasting for tag: {tag}")
        
        # Filter tag
        tag_data = trends_df.filter(F.col("Tag") == tag).orderBy("Period")
        
        # Get historical data
        history = tag_data.select("Period", "QuestionCount").collect()
        
        if len(history) < 3:
            logger.warning(f"Not enough data for tag: {tag}")
            return self.spark.createDataFrame([], "Period: date, Forecast: double")
        
        # Simple exponential smoothing
        alpha = 0.3
        values = [row["QuestionCount"] for row in history]
        
        # Initialize
        smoothed = [values[0]]
        for i in range(1, len(values)):
            smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[-1])
        
        # Forecast
        last_smoothed = smoothed[-1]
        last_period = history[-1]["Period"]
        
        forecasts = []
        for i in range(1, periods + 1):
            # Simple: forecast is last smoothed value
            # In production, use proper time series methods
            forecast_value = last_smoothed
            forecasts.append({
                "Period": last_period,  # Placeholder
                "Forecast": float(forecast_value),
                "Tag": tag
            })
        
        return self.spark.createDataFrame(forecasts)
    
    def compare_tags(
        self,
        trends_df: DataFrame,
        tags: List[str]
    ) -> DataFrame:
        """
        Membandingkan tren beberapa tags.
        
        Parameters
        ----------
        trends_df : DataFrame
            Trend data
        tags : List[str]
            Tags untuk dibandingkan
            
        Returns
        -------
        DataFrame
            Comparison data
        """
        # Filter tags
        filtered = trends_df.filter(F.col("Tag").isin(tags))
        
        # Pivot untuk easy comparison
        pivoted = filtered.groupBy("Period").pivot("Tag").agg(
            F.first("QuestionCount")
        ).orderBy("Period")
        
        return pivoted
    
    def get_top_tags_by_period(
        self,
        trends_df: DataFrame,
        n: int = 10
    ) -> DataFrame:
        """
        Mendapatkan top N tags per periode.
        
        Parameters
        ----------
        trends_df : DataFrame
            Trend data
        n : int
            Jumlah top tags
            
        Returns
        -------
        DataFrame
            Top tags per period
        """
        window_spec = Window.partitionBy("Period").orderBy(F.desc("QuestionCount"))
        
        ranked = trends_df.withColumn("Rank", F.row_number().over(window_spec))
        top_n = ranked.filter(F.col("Rank") <= n)
        
        return top_n
    
    def calculate_yoy_growth(
        self,
        trends_df: DataFrame
    ) -> DataFrame:
        """
        Menghitung Year-over-Year growth.
        
        Parameters
        ----------
        trends_df : DataFrame
            Trend data dengan granularitas monthly atau yearly
            
        Returns
        -------
        DataFrame
            YoY growth per tag
        """
        # Add year column
        with_year = trends_df.withColumn("Year", F.year("Period"))
        
        # Aggregate by year
        yearly = with_year.groupBy("Tag", "Year").agg(
            F.sum("QuestionCount").alias("YearlyCount")
        )
        
        # Calculate YoY
        window_spec = Window.partitionBy("Tag").orderBy("Year")
        
        yoy = yearly.withColumn(
            "PrevYearCount",
            F.lag("YearlyCount", 1).over(window_spec)
        ).withColumn(
            "YoYGrowth",
            (F.col("YearlyCount") - F.col("PrevYearCount")) /
            F.greatest(F.col("PrevYearCount"), F.lit(1)) * 100
        )
        
        return yoy
    
    def identify_emerging_technologies(
        self,
        trends_df: DataFrame,
        min_growth: float = 50.0,
        min_questions: int = 50
    ) -> DataFrame:
        """
        Mengidentifikasi teknologi emerging.
        
        Parameters
        ----------
        trends_df : DataFrame
            Trend data
        min_growth : float
            Minimum growth rate (%)
        min_questions : int
            Minimum pertanyaan
            
        Returns
        -------
        DataFrame
            Emerging technologies
        """
        logger.info("Identifying emerging technologies")
        
        yoy = self.calculate_yoy_growth(trends_df)
        
        # Get latest year
        latest_year = yoy.agg(F.max("Year")).collect()[0][0]
        
        emerging = yoy \
            .filter(F.col("Year") == latest_year) \
            .filter(F.col("YoYGrowth") >= min_growth) \
            .filter(F.col("YearlyCount") >= min_questions) \
            .orderBy(F.desc("YoYGrowth"))
        
        return emerging
