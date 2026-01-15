"""
ML Module for Stack Overflow Analytics
======================================
Module untuk Machine Learning dan Predictive Analytics.
"""

from .quality_predictor import QualityPredictor
from .trend_forecaster import TrendForecaster
from .duplicate_detector import DuplicateDetector

__all__ = ['QualityPredictor', 'TrendForecaster', 'DuplicateDetector']
