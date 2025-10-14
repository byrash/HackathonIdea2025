"""
Fraud Detection Services Package

Contains all analysis services for bank check fraud detection.
"""

from .fraud_detector import FraudDetector
from .image_processor import ImageProcessor
from .metadata_analyzer import MetadataAnalyzer
from .ocr_service import OCRService
from .forensic_analyzer import ForensicAnalyzer
from .template_matcher import TemplateMatcher
from .ml_predictor import MLPredictor

__all__ = [
    'FraudDetector',
    'ImageProcessor',
    'MetadataAnalyzer',
    'OCRService',
    'ForensicAnalyzer',
    'TemplateMatcher',
    'MLPredictor',
]

