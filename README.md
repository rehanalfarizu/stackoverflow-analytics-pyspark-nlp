# Stack Overflow Analytics with PySpark and NLP

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/PySpark-3.5.0-orange.svg)](https://spark.apache.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Big Data](https://img.shields.io/badge/Big%20Data-Analytics-red.svg)](#)
[![NLP](https://img.shields.io/badge/NLP-NLTK-purple.svg)](#)
[![ML](https://img.shields.io/badge/ML-Spark%20MLlib-yellow.svg)](#)

---

## Overview

A comprehensive Big Data analytics platform for analyzing **Stack Overflow Data Dump** using **Apache Spark (PySpark)** and **Natural Language Processing (NLP)** techniques. This project demonstrates end-to-end data pipeline from ETL processing to machine learning predictions.

---

## Features

- **ETL Pipeline** - XML parsing, data transformation, and Parquet storage
- **NLP Processing** - Text preprocessing, TF-IDF, sentiment analysis, topic modeling
- **Machine Learning** - Question quality prediction, trend forecasting, duplicate detection
- **Visualization** - Interactive Streamlit dashboard with real-time analytics
- **Scalable Architecture** - Designed for distributed processing with Apache Spark

---

## Dataset

Data sourced from [Stack Exchange Data Dump](https://archive.org/details/stackexchange):

| File | Description | Size |
|------|-------------|------|
| Posts.xml | Questions and answers | ~90 GB |
| Users.xml | User information | ~3 GB |
| Comments.xml | Comments data | ~20 GB |
| Tags.xml | Tag definitions | ~5 MB |
| Votes.xml | Voting data | ~15 GB |

---

## Architecture

```
+------------------+     +------------------+     +------------------+
|   Data Source    | --> |   ETL Pipeline   | --> |   Data Storage   |
|  (XML/Parquet)   |     |   (PySpark)      |     |   (Parquet)      |
+------------------+     +------------------+     +------------------+
                                  |
                                  v
+------------------+     +------------------+     +------------------+
|   NLP Module     | <-- |   Processing     | --> |   ML Module      |
| (NLTK/TF-IDF)    |     |   (Spark SQL)    |     | (MLlib/CV)       |
+------------------+     +------------------+     +------------------+
                                  |
                                  v
                         +------------------+
                         |   Visualization  |
                         |   (Streamlit)    |
                         +------------------+
```

---

## Project Structure

```
stackoverflow-analytics-pyspark-nlp/
|-- config/
|   |-- settings.yaml
|   |-- spark_config.py
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- output/
|-- notebooks/
|   |-- stackoverflow_analytics_colab.ipynb
|   |-- requirements_compliance.ipynb
|-- scripts/
|   |-- download_data.sh
|   |-- run_pipeline.sh
|-- src/
|   |-- etl/
|   |   |-- xml_parser.py
|   |   |-- data_transformer.py
|   |   |-- data_loader.py
|   |   |-- rdd_ops.py
|   |-- nlp/
|   |   |-- text_preprocessor.py
|   |   |-- feature_extractor.py
|   |   |-- topic_modeler.py
|   |   |-- sentiment_analyzer.py
|   |-- ml/
|   |   |-- quality_predictor.py
|   |   |-- trend_forecaster.py
|   |   |-- duplicate_detector.py
|   |-- utils/
|   |   |-- helpers.py
|   |   |-- logger.py
|   |-- visualization/
|       |-- dashboard.py
|       |-- charts.py
|-- tests/
|-- main.py
|-- Makefile
|-- requirements.txt
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/username/stackoverflow-analytics-pyspark-nlp.git
cd stackoverflow-analytics-pyspark-nlp

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Command Line

```bash
python main.py --mode full      # Full pipeline
python main.py --mode etl       # ETL only
python main.py --mode nlp       # NLP only
python main.py --mode ml        # ML only
make dashboard                  # Streamlit dashboard
```

### Jupyter / Google Colab

```bash
jupyter notebook notebooks/
```

---

## Machine Learning Models

### Algorithms

| Model | Parameters |
|-------|------------|
| Random Forest | numTrees: [10, 20, 50], maxDepth: [5, 10] |
| Logistic Regression | regParam: [0.01, 0.1], elasticNetParam: [0.0, 0.5] |

### Evaluation Metrics

| Metric | Random Forest | Logistic Regression |
|--------|---------------|---------------------|
| Accuracy | 1.0000 | 1.0000 |
| F1-Score | 1.0000 | 1.0000 |
| Precision | 1.0000 | 1.0000 |
| Recall | 1.0000 | 1.0000 |
| AUC-ROC | 1.0000 | 1.0000 |

---

## RDD Operations

| Operation | Description |
|-----------|-------------|
| map / flatMap | Data transformation and tag extraction |
| reduceByKey | Count aggregation |
| groupByKey | Post ID grouping |
| combineByKey | Statistics calculation (min/max/sum) |
| aggregateByKey | Average score computation |
| partitionBy | Custom data partitioning |

---

## Spark SQL Features

```sql
-- CTE (Common Table Expression)
WITH high_score_posts AS (
    SELECT * FROM posts WHERE Score > 10
)
SELECT Tags, COUNT(*) FROM high_score_posts GROUP BY Tags

-- Subquery
SELECT * FROM posts WHERE Score > (SELECT AVG(Score) FROM posts)

-- Broadcast Hint
SELECT /*+ BROADCAST(tags) */ p.*, t.TagName
FROM posts p JOIN tags t ON p.Tags LIKE CONCAT('%', t.TagName, '%')
```

---

## Testing

```bash
pytest tests/ -v
```

---

## Tech Stack

- **Processing**: Apache Spark, PySpark
- **NLP**: NLTK, Spark NLP
- **ML**: Spark MLlib, CrossValidator
- **Storage**: Parquet, HDFS
- **Visualization**: Streamlit, Plotly, Matplotlib
- **Testing**: Pytest

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Author

**Muhammad Raihan Alfarizi**  
NIM: 23.11.5548

Big Data Predictive Analytics  
Universitas Amikom Yogyakarta

---

## References

1. [Stack Exchange Data Dump](https://archive.org/details/stackexchange)
2. [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
3. [PySpark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
4. [NLTK Documentation](https://www.nltk.org/)
