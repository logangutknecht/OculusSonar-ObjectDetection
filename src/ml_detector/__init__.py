"""
Machine Learning Detector Module for Sonar Object Detection
"""

from .ml_sonar_detector import MLSonarDetector, HybridSonarDetector
from .train_sonar_detector import SonarModelTrainer, train_from_scratch
from .data_annotator import SonarAnnotator, BoundingBox
from .validate_model import ModelValidator, PerformanceAnalyzer

__all__ = [
    'MLSonarDetector',
    'HybridSonarDetector', 
    'SonarModelTrainer',
    'train_from_scratch',
    'SonarAnnotator',
    'BoundingBox',
    'ModelValidator',
    'PerformanceAnalyzer'
]

__version__ = '1.0.0'
