"""
Federated Learning for Fraud Detection
A comprehensive framework for privacy-preserving fraud detection across multiple institutions.
"""

__version__ = "2.0.0"
__author__ = "Enhanced by ML Team"

from .models import FraudDetectionModel
from .privacy import DifferentialPrivacyEngine
from .utils import DataPreprocessor
from .aggregators import FedAvgAggregator, FedProxAggregator, FedDANEAggregator

__all__ = [
    "FraudDetectionModel",
    "DifferentialPrivacyEngine",
    "DataPreprocessor",
    "FedAvgAggregator",
    "FedProxAggregator",
    "FedDANEAggregator",
]
