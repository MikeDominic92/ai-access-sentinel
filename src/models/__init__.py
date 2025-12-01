"""ML models for IAM analysis."""

from .anomaly_detector import AnomalyDetector, EnsembleAnomalyDetector
from .access_predictor import AccessPredictor
from .role_miner import RoleMiner
from .risk_scorer import RiskScorer

# Deep learning models (optional, requires TensorFlow)
try:
    from .lstm_detector import LSTMDetector
    from .transformer_detector import TransformerDetector
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    LSTMDetector = None
    TransformerDetector = None

__all__ = [
    'AnomalyDetector',
    'EnsembleAnomalyDetector',
    'AccessPredictor',
    'RoleMiner',
    'RiskScorer',
    'LSTMDetector',
    'TransformerDetector',
    'DEEP_LEARNING_AVAILABLE'
]
