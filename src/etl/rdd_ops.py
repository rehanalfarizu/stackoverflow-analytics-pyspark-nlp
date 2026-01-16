"""
RDD Operations and MapReduce Examples
=====================================
Implementasi operasi RDD: map, flatMap, partitionBy, reduceByKey,
 groupByKey, combineByKey, aggregateByKey untuk dataset Stack Overflow.
"""

from typing import List, Tuple
import re
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType


def _extract_tags(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"<([^>]+)>", text)


def tags_count_mapreduce(df: DataFrame) -> DataFrame:
    """Hitung jumlah pertanyaan per tag menggunakan MapReduce (RDD)."""
    rdd = (
        df.select("Tags")
        .where(F.col("Tags").isNotNull())
        .rdd
        .flatMap(lambda row: _extract_tags(row["Tags"]))
        .map(lambda tag: (tag, 1))
        .reduceByKey(lambda a, b: a + b)
        .sortBy(lambda kv: -kv[1])
    )

    schema = StructType([
        StructField("Tag", StringType(), True),
        StructField("Count", IntegerType(), True),
    ])
    return df.sql_ctx.createDataFrame(rdd, schema)


def monthly_question_mapreduce(df: DataFrame, partitions: int = 4) -> DataFrame:
    """Hitung jumlah pertanyaan per (Year, Month) menggunakan MapReduce dengan partitionBy."""
    rdd = (
        df.select("Id", "Year", "Month")
        .where(F.col("Year").isNotNull() & F.col("Month").isNotNull())
        .rdd
        .map(lambda row: ((row["Year"], row["Month"]), 1))
        .partitionBy(partitions)
        .reduceByKey(lambda a, b: a + b)
        .sortBy(lambda kv: (kv[0][0], kv[0][1]))
    )

    schema = StructType([
        StructField("Year", IntegerType(), True),
        StructField("Month", IntegerType(), True),
        StructField("QuestionCount", IntegerType(), True),
    ])
    rows = rdd.map(lambda kv: (kv[0][0], kv[0][1], kv[1]))
    return df.sql_ctx.createDataFrame(rows, schema)


def tag_score_aggregate(df: DataFrame) -> DataFrame:
    """AggregateByKey untuk menghitung total skor dan rata-rata skor per tag."""
    rdd = (
        df.select("Tags", "Score")
        .where(F.col("Tags").isNotNull() & F.col("Score").isNotNull())
        .rdd
        .flatMap(lambda row: [(tag, row["Score"]) for tag in _extract_tags(row["Tags"])])
        .aggregateByKey(
            (0, 0),  # (sum, count)
            lambda acc, v: (acc[0] + v, acc[1] + 1),
            lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1]),
        )
        .map(lambda kv: (kv[0], kv[1][0], kv[1][1], (kv[1][0] / kv[1][1]) if kv[1][1] > 0 else 0))
        .sortBy(lambda x: -x[3])
    )

    schema = StructType([
        StructField("Tag", StringType(), True),
        StructField("TotalScore", IntegerType(), True),
        StructField("Count", IntegerType(), True),
        StructField("AvgScore", IntegerType(), True),
    ])
    # Cast AvgScore to int for schema consistency
    rows = rdd.map(lambda x: (x[0], int(x[1]), int(x[2]), int(x[3])))
    return df.sql_ctx.createDataFrame(rows, schema)


def tag_posts_groupbykey(df: DataFrame) -> DataFrame:
    """Contoh groupByKey untuk mengumpulkan Id per tag (hanya untuk demonstrasi)."""
    rdd = (
        df.select("Id", "Tags")
        .where(F.col("Tags").isNotNull())
        .rdd
        .flatMap(lambda row: [(tag, row["Id"]) for tag in _extract_tags(row["Tags"])])
        .groupByKey()
        .map(lambda kv: (kv[0], len(list(kv[1]))))
        .sortBy(lambda x: -x[1])
    )

    schema = StructType([
        StructField("Tag", StringType(), True),
        StructField("PostIdsCount", IntegerType(), True),
    ])
    return df.sql_ctx.createDataFrame(rdd, schema)


def tag_stats_combinebykey(df: DataFrame) -> DataFrame:
    """Contoh combineByKey untuk menghitung (min, max, sum) skor per tag."""
    def create_comb(v: int):
        return (v, v, v, 1)  # (min, max, sum, count)

    def merge_val(acc, v: int):
        return (min(acc[0], v), max(acc[1], v), acc[2] + v, acc[3] + 1)

    def merge_comb(acc1, acc2):
        return (min(acc1[0], acc2[0]), max(acc1[1], acc2[1]), acc1[2] + acc2[2], acc1[3] + acc2[3])

    rdd = (
        df.select("Tags", "Score")
        .where(F.col("Tags").isNotNull() & F.col("Score").isNotNull())
        .rdd
        .flatMap(lambda row: [(tag, row["Score"]) for tag in _extract_tags(row["Tags"])])
        .combineByKey(create_comb, merge_val, merge_comb)
        .map(lambda kv: (kv[0], kv[1][0], kv[1][1], kv[1][2], kv[1][3]))
        .sortBy(lambda x: -x[4])
    )

    schema = StructType([
        StructField("Tag", StringType(), True),
        StructField("MinScore", IntegerType(), True),
        StructField("MaxScore", IntegerType(), True),
        StructField("SumScore", IntegerType(), True),
        StructField("Count", IntegerType(), True),
    ])
    return df.sql_ctx.createDataFrame(rdd, schema)


def demo_all(df: DataFrame, spark: SparkSession) -> None:
    """Jalankan semua demo operasi RDD dan tampilkan hasilnya."""
    print("[RDD] MapReduce: Tag counts")
    tags_df = tags_count_mapreduce(df)
    tags_df.show(10, truncate=False)

    print("[RDD] Monthly counts with partitionBy + reduceByKey")
    monthly_df = monthly_question_mapreduce(df)
    monthly_df.show(12, truncate=False)

    print("[RDD] AggregateByKey: Avg score per tag")
    agg_df = tag_score_aggregate(df)
    agg_df.show(10, truncate=False)

    print("[RDD] GroupByKey: Post IDs count per tag")
    gpk_df = tag_posts_groupbykey(df)
    gpk_df.show(10, truncate=False)

    print("[RDD] CombineByKey: Stats per tag")
    cbk_df = tag_stats_combinebykey(df)
    cbk_df.show(10, truncate=False)

    # Example of partition information
    rdd = df.rdd
    print(f"[RDD] Partitions: {rdd.getNumPartitions()}")
