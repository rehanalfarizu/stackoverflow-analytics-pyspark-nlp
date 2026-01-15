"""
Logger Module
=============
Konfigurasi logging untuk aplikasi.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
import os
from typing import Optional


def setup_logger(
    name: str = "stackoverflow_analytics",
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup logger dengan konfigurasi standar.
    
    Parameters
    ----------
    name : str
        Nama logger
    level : str
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str, optional
        Path file log
    max_size : int
        Ukuran maksimum file log (bytes)
    backup_count : int
        Jumlah backup files
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "stackoverflow_analytics") -> logging.Logger:
    """
    Mendapatkan logger yang sudah ada atau membuat baru.
    
    Parameters
    ----------
    name : str
        Nama logger
        
    Returns
    -------
    logging.Logger
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If no handlers, setup default
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


class LogContext:
    """Context manager untuk logging dengan konteks."""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is not None:
            self.logger.error(
                f"Failed: {self.operation} after {duration:.2f}s - {exc_val}"
            )
        else:
            self.logger.info(
                f"Completed: {self.operation} in {duration:.2f}s"
            )
        
        return False  # Don't suppress exceptions


# Predefined loggers
def get_etl_logger() -> logging.Logger:
    """Get ETL-specific logger."""
    return get_logger("stackoverflow_analytics.etl")


def get_nlp_logger() -> logging.Logger:
    """Get NLP-specific logger."""
    return get_logger("stackoverflow_analytics.nlp")


def get_ml_logger() -> logging.Logger:
    """Get ML-specific logger."""
    return get_logger("stackoverflow_analytics.ml")
