"""
Stack Overflow Analytics - Main Entry Point
============================================
Entry point utama untuk menjalankan pipeline analytics.
"""

import argparse
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.spark_config import create_spark_session
from src.utils.logger import setup_logger
from src.utils.helpers import Timer, load_config


def run_etl(spark, config, logger):
    """Run ETL pipeline."""
    from src.etl import XMLParser, DataTransformer, DataLoader
    
    logger.info("Starting ETL Pipeline...")
    
    raw_dir = config['paths']['raw_data']
    processed_dir = config['paths']['processed_data']
    
    # Initialize components
    parser = XMLParser(spark)
    transformer = DataTransformer(spark)
    loader = DataLoader(spark)
    
    # Check for data files
    posts_file = os.path.join(raw_dir, "Posts.xml")
    
    if os.path.exists(posts_file):
        logger.info("Parsing Posts.xml...")
        posts_df = parser.parse_posts(posts_file)
        
        logger.info("Transforming data...")
        posts_df = transformer.clean_posts(posts_df)
        
        # Filter questions
        questions_df = transformer.filter_questions(posts_df)
        
        logger.info("Saving processed data...")
        loader.save_parquet(
            questions_df,
            os.path.join(processed_dir, "questions"),
            partition_by=["Year", "Month"]
        )
        
        logger.info(f"ETL completed. Processed {questions_df.count()} questions.")
        return questions_df
    else:
        logger.warning(f"Posts.xml not found in {raw_dir}")
        logger.info("Creating sample data for demonstration...")
        
        # Create sample data
        sample_data = [
            (1, 1, "How to read CSV in Python pandas?", 
             "<p>I want to read a CSV file using pandas.</p>", 
             "<python><pandas><csv>", 10, 1500, 3, 1, "2024-01-15"),
            (2, 1, "JavaScript async await tutorial",
             "<p>Can someone explain async/await in JavaScript?</p>",
             "<javascript><async><promise>", 25, 3200, 5, 1, "2024-02-20"),
            (3, 1, "Docker container networking",
             "<p>How to connect containers in Docker?</p>",
             "<docker><networking><containers>", 15, 2100, 4, 1, "2024-03-10"),
        ]
        
        sample_df = spark.createDataFrame(
            sample_data,
            ["Id", "PostTypeId", "Title", "Body", "Tags", "Score", 
             "ViewCount", "AnswerCount", "HasAcceptedAnswer", "CreationDate"]
        )
        
        logger.info("Sample data created.")
        return sample_df


def run_nlp(spark, df, config, logger):
    """Run NLP analysis."""
    from src.nlp import TextPreprocessor, TopicModeler, SentimentAnalyzer
    
    logger.info("Starting NLP Analysis...")
    
    # Preprocess
    preprocessor = TextPreprocessor(spark)
    processed_df = preprocessor.preprocess(df, text_column="Body" if "Body" in df.columns else "Title")
    
    logger.info("Text preprocessing completed.")
    
    # Topic Modeling (if enough data)
    if processed_df.count() >= 10:
        logger.info("Running topic modeling...")
        modeler = TopicModeler(spark)
        try:
            model, topics_df = modeler.fit_lda(processed_df, num_topics=5, max_iterations=10)
            modeler.print_topics(num_words=5)
        except Exception as e:
            logger.warning(f"Topic modeling skipped: {e}")
    
    logger.info("NLP Analysis completed.")
    return processed_df


def run_ml(spark, df, config, logger):
    """Run ML training."""
    from src.ml import QualityPredictor, TrendForecaster
    
    logger.info("Starting ML Training...")
    
    # Quality Prediction
    predictor = QualityPredictor(spark)
    
    # Prepare data
    if "Score" in df.columns:
        from pyspark.sql import functions as F
        
        # Create quality labels
        df = df.withColumn(
            "QualityLabel",
            F.when(F.col("Score") >= 10, 2)
            .when(F.col("Score") >= 0, 1)
            .otherwise(0)
        )
        
        # Add basic features
        df = df.withColumn("TitleLength", F.length("Title"))
        
        if "Body" in df.columns:
            df = df.withColumn("BodyLength", F.length("Body"))
        else:
            df = df.withColumn("BodyLength", F.lit(0))
        
        df = df.fillna(0)
        
        logger.info("Training quality predictor...")
        try:
            metrics = predictor.train(
                df,
                algorithm="random_forest",
                feature_columns=["TitleLength", "BodyLength"]
            )
            logger.info(f"Model trained. Accuracy: {metrics.get('accuracy', 'N/A')}")
        except Exception as e:
            logger.warning(f"ML training skipped: {e}")
    
    logger.info("ML Training completed.")
    return df


def run_dashboard(config, logger):
    """Run Streamlit dashboard."""
    logger.info("Starting Dashboard...")
    
    import subprocess
    dashboard_path = "src/visualization/dashboard.py"
    
    if os.path.exists(dashboard_path):
        subprocess.run(["streamlit", "run", dashboard_path])
    else:
        logger.error(f"Dashboard file not found: {dashboard_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Stack Overflow Analytics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode full          Run full pipeline
  python main.py --mode etl           Run only ETL
  python main.py --mode nlp           Run only NLP analysis
  python main.py --mode ml            Run only ML training
  python main.py --mode dashboard     Launch dashboard
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'etl', 'nlp', 'ml', 'dashboard'],
        default='full',
        help='Pipeline mode to run'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/settings.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger(
        name="stackoverflow_analytics",
        level=args.log_level,
        log_file="logs/app.log"
    )
    
    logger.info("="*60)
    logger.info("Stack Overflow Analytics Pipeline")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("="*60)
    
    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {args.config}")
        config = {
            'paths': {
                'raw_data': 'data/raw',
                'processed_data': 'data/processed',
                'output_data': 'data/output'
            }
        }
    
    # Ensure directories exist
    for path in config['paths'].values():
        os.makedirs(path, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Dashboard mode doesn't need Spark
    if args.mode == 'dashboard':
        run_dashboard(config, logger)
        return
    
    # Create Spark session
    with Timer("Spark Session Creation"):
        spark = create_spark_session(app_name="StackOverflowAnalytics")
    
    try:
        df = None
        
        # Run pipeline based on mode
        if args.mode in ['full', 'etl']:
            with Timer("ETL Pipeline"):
                df = run_etl(spark, config, logger)
        
        if args.mode in ['full', 'nlp']:
            if df is None:
                # Load processed data
                processed_path = os.path.join(config['paths']['processed_data'], "questions")
                if os.path.exists(processed_path):
                    df = spark.read.parquet(processed_path)
                else:
                    df = run_etl(spark, config, logger)
            
            with Timer("NLP Analysis"):
                df = run_nlp(spark, df, config, logger)
        
        if args.mode in ['full', 'ml']:
            if df is None:
                processed_path = os.path.join(config['paths']['processed_data'], "questions")
                if os.path.exists(processed_path):
                    df = spark.read.parquet(processed_path)
                else:
                    df = run_etl(spark, config, logger)
            
            with Timer("ML Training"):
                df = run_ml(spark, df, config, logger)
        
        logger.info("="*60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Finished at: {datetime.now()}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    
    finally:
        spark.stop()
        logger.info("Spark session stopped.")


if __name__ == "__main__":
    main()
