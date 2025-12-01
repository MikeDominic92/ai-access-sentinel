"""
Deep learning models for IAM anomaly detection.

Provides advanced neural network architectures for temporal and contextual
anomaly detection in identity and access management.
"""

from typing import Optional

# Import models with error handling for TensorFlow dependency
try:
    from ..lstm_detector import LSTMDetector
    from ..transformer_detector import TransformerDetector
    DEEP_LEARNING_AVAILABLE = True
except ImportError as e:
    DEEP_LEARNING_AVAILABLE = False
    import warnings
    warnings.warn(
        f"Deep learning models unavailable: {e}\n"
        "Install TensorFlow: pip install tensorflow>=2.13.0"
    )

    # Create placeholder classes
    class LSTMDetector:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow required. Install with: pip install tensorflow>=2.13.0")

    class TransformerDetector:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow required. Install with: pip install tensorflow>=2.13.0")


__all__ = [
    'LSTMDetector',
    'TransformerDetector',
    'DEEP_LEARNING_AVAILABLE'
]


def get_detector(model_type: str, **kwargs):
    """
    Factory function to create deep learning detectors.

    Args:
        model_type: Type of detector ('lstm' or 'transformer')
        **kwargs: Arguments to pass to detector constructor

    Returns:
        Detector instance

    Raises:
        ValueError: If model_type is unknown
        ImportError: If TensorFlow is not available
    """
    if not DEEP_LEARNING_AVAILABLE:
        raise ImportError(
            "Deep learning models require TensorFlow.\n"
            "Install with: pip install tensorflow>=2.13.0"
        )

    model_type = model_type.lower()

    if model_type == 'lstm':
        return LSTMDetector(**kwargs)
    elif model_type == 'transformer':
        return TransformerDetector(**kwargs)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose 'lstm' or 'transformer'"
        )


# Model selection guidance
MODEL_SELECTION_GUIDE = """
Deep Learning Model Selection Guide
====================================

LSTM Detector
-------------
Best for:
- Sequential access patterns where order matters
- Multi-step attack detection (reconnaissance -> access -> exfiltration)
- Time-series analysis of user behavior
- Detecting gradual privilege escalation

Use when:
- You have temporal/sequential data
- Attack patterns span multiple events
- Order of access is important
- You want to detect multi-stage attacks

Example: Detecting an attacker who first explores the system (reconnaissance),
then accesses sensitive data (breach), then attempts to download (exfiltration)


Transformer Detector
--------------------
Best for:
- Feature importance analysis (which features matter most)
- Context-aware single-event anomaly detection
- Understanding relationships between features
- Tabular IAM data without strong temporal ordering

Use when:
- You want to know WHY an event is anomalous
- Feature relationships are important
- You don't need sequential context
- You want attention-based interpretability

Example: Detecting that an access is anomalous because it combines unusual
location + unusual time + sensitive resource (attention shows which mattered most)


Comparison with Traditional ML
-------------------------------
                 | Isolation Forest | LSTM           | Transformer
-----------------+------------------+----------------+------------------
Data Type        | Tabular          | Sequential     | Tabular
Training Time    | Fast (seconds)   | Slow (minutes) | Medium (minutes)
Inference Speed  | Very Fast        | Fast           | Fast
Interpretability | Medium           | Low            | Medium-High
Accuracy         | Good (~87%)      | Better (~92%)  | Better (~90%)
Data Required    | Less (100s)      | More (1000s)   | More (1000s)
Use Case         | Quick baseline   | Attack chains  | Feature analysis

Recommendation: Start with Isolation Forest for quick baseline, then add
LSTM for temporal patterns or Transformer for better interpretability.
"""


def print_selection_guide():
    """Print model selection guide."""
    print(MODEL_SELECTION_GUIDE)


if __name__ == "__main__":
    print_selection_guide()
