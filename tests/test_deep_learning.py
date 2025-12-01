"""
Unit tests for deep learning models.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import deep learning models
try:
    from src.models.lstm_detector import LSTMDetector
    from src.models.transformer_detector import TransformerDetector
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TENSORFLOW_AVAILABLE,
    reason="TensorFlow not installed"
)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    # Normal samples
    X_normal = np.random.randn(150, n_features) * 0.5
    y_normal = np.zeros(150)

    # Anomalous samples
    X_anomaly = np.random.randn(50, n_features) * 2.0 + 3.0
    y_anomaly = np.ones(50)

    # Combine and shuffle
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([y_normal, y_anomaly])

    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    feature_names = [f'feature_{i}' for i in range(n_features)]

    return X, y, feature_names


class TestLSTMDetector:
    """Tests for LSTM anomaly detector."""

    def test_initialization(self):
        """Test LSTM detector initialization."""
        detector = LSTMDetector(
            sequence_length=10,
            n_features=5,
            lstm_units=[64, 32],
            dropout_rate=0.2
        )

        assert detector.sequence_length == 10
        assert detector.n_features == 5
        assert detector.lstm_units == [64, 32]
        assert detector.dropout_rate == 0.2
        assert not detector.is_trained

    def test_train(self, sample_data):
        """Test LSTM training."""
        X, y, feature_names = sample_data

        detector = LSTMDetector(
            sequence_length=5,
            n_features=X.shape[1]
        )

        # Train
        detector.train(
            X, y,
            feature_names=feature_names,
            epochs=2,  # Quick test
            verbose=0
        )

        assert detector.is_trained
        assert detector.model is not None
        assert detector.feature_names == feature_names

    def test_predict(self, sample_data):
        """Test LSTM prediction."""
        X, y, feature_names = sample_data

        detector = LSTMDetector(sequence_length=5)
        detector.train(X, y, epochs=2, verbose=0)

        # Predict
        probabilities = detector.predict(X)

        assert len(probabilities) > 0
        assert all(0 <= p <= 1 for p in probabilities)

    def test_predict_classes(self, sample_data):
        """Test LSTM class prediction."""
        X, y, feature_names = sample_data

        detector = LSTMDetector(sequence_length=5, threshold=0.5)
        detector.train(X, y, epochs=2, verbose=0)

        # Predict classes
        predictions = detector.predict_classes(X)

        assert len(predictions) > 0
        assert all(p in [0, 1] for p in predictions)

    def test_analyze_sequence(self, sample_data):
        """Test sequence analysis."""
        X, y, feature_names = sample_data

        detector = LSTMDetector(sequence_length=10)
        detector.train(X, y, epochs=2, verbose=0)

        # Analyze sequence
        sequence = X[:10]
        result = detector.analyze_sequence(sequence)

        assert 'is_anomaly' in result
        assert 'anomaly_probability' in result
        assert 'risk_level' in result
        assert 'risk_score' in result
        assert result['detector'] == 'LSTM'
        assert isinstance(result['is_anomaly'], bool)
        assert 0 <= result['anomaly_probability'] <= 1

    def test_detect_attack_patterns(self, sample_data):
        """Test attack pattern detection."""
        X, y, feature_names = sample_data

        detector = LSTMDetector(sequence_length=5)
        detector.train(X, y, epochs=2, verbose=0)

        # Detect patterns
        user_ids = np.array([f'U{i:03d}' for i in range(len(X))])
        patterns = detector.detect_attack_patterns(X, user_ids)

        assert 'sequence_start' in patterns.columns
        assert 'sequence_end' in patterns.columns
        assert 'anomaly_probability' in patterns.columns
        assert 'pattern_type' in patterns.columns

    def test_evaluate(self, sample_data):
        """Test model evaluation."""
        X, y, feature_names = sample_data

        detector = LSTMDetector(sequence_length=5)
        detector.train(X, y, epochs=2, verbose=0)

        # Evaluate
        metrics = detector.evaluate(X, y)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert all(0 <= v <= 1 for v in metrics.values())

    def test_save_load(self, sample_data, tmp_path):
        """Test model saving and loading."""
        X, y, feature_names = sample_data

        detector = LSTMDetector(sequence_length=5)
        detector.train(X, y, feature_names=feature_names, epochs=2, verbose=0)

        # Save
        save_path = str(tmp_path / "lstm_test")
        detector.save(save_path)

        # Load
        detector2 = LSTMDetector()
        detector2.load(save_path)

        assert detector2.is_trained
        assert detector2.sequence_length == detector.sequence_length
        assert detector2.n_features == detector.n_features
        assert detector2.feature_names == detector.feature_names

        # Predictions should be similar
        pred1 = detector.predict(X[:20])
        pred2 = detector2.predict(X[:20])
        np.testing.assert_allclose(pred1, pred2, rtol=1e-5)

    def test_untrained_error(self, sample_data):
        """Test error when using untrained model."""
        X, y, feature_names = sample_data

        detector = LSTMDetector(sequence_length=5)

        with pytest.raises(ValueError, match="not trained"):
            detector.predict(X)


class TestTransformerDetector:
    """Tests for Transformer anomaly detector."""

    def test_initialization(self):
        """Test Transformer detector initialization."""
        detector = TransformerDetector(
            n_features=10,
            embed_dim=32,
            num_heads=4,
            ff_dim=64
        )

        assert detector.n_features == 10
        assert detector.embed_dim == 32
        assert detector.num_heads == 4
        assert detector.ff_dim == 64
        assert not detector.is_trained

    def test_train(self, sample_data):
        """Test Transformer training."""
        X, y, feature_names = sample_data

        detector = TransformerDetector(
            n_features=X.shape[1],
            embed_dim=16,
            num_heads=2
        )

        # Train
        detector.train(
            X, y,
            feature_names=feature_names,
            epochs=2,  # Quick test
            verbose=0
        )

        assert detector.is_trained
        assert detector.model is not None
        assert detector.feature_names == feature_names

    def test_predict(self, sample_data):
        """Test Transformer prediction."""
        X, y, feature_names = sample_data

        detector = TransformerDetector(n_features=X.shape[1])
        detector.train(X, y, epochs=2, verbose=0)

        # Predict
        probabilities = detector.predict(X)

        assert len(probabilities) == len(X)
        assert all(0 <= p <= 1 for p in probabilities)

    def test_predict_classes(self, sample_data):
        """Test Transformer class prediction."""
        X, y, feature_names = sample_data

        detector = TransformerDetector(n_features=X.shape[1], threshold=0.5)
        detector.train(X, y, epochs=2, verbose=0)

        # Predict classes
        predictions = detector.predict_classes(X)

        assert len(predictions) == len(X)
        assert all(p in [0, 1] for p in predictions)

    def test_analyze_event_array(self, sample_data):
        """Test event analysis with numpy array."""
        X, y, feature_names = sample_data

        detector = TransformerDetector(n_features=X.shape[1])
        detector.train(X, y, feature_names=feature_names, epochs=2, verbose=0)

        # Analyze event
        sample = X[0]
        result = detector.analyze_event(sample)

        assert 'is_anomaly' in result
        assert 'anomaly_probability' in result
        assert 'risk_level' in result
        assert 'risk_score' in result
        assert result['detector'] == 'Transformer'
        assert isinstance(result['is_anomaly'], bool)

    def test_analyze_event_dict(self, sample_data):
        """Test event analysis with dictionary."""
        X, y, feature_names = sample_data

        detector = TransformerDetector(n_features=X.shape[1])
        detector.train(X, y, feature_names=feature_names, epochs=2, verbose=0)

        # Create feature dict
        sample_dict = {name: float(X[0, i]) for i, name in enumerate(feature_names)}
        result = detector.analyze_event(sample_dict)

        assert 'is_anomaly' in result
        assert 'top_features' in result
        assert len(result['top_features']) > 0

    def test_evaluate(self, sample_data):
        """Test model evaluation."""
        X, y, feature_names = sample_data

        detector = TransformerDetector(n_features=X.shape[1])
        detector.train(X, y, epochs=2, verbose=0)

        # Evaluate
        metrics = detector.evaluate(X, y)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert all(0 <= v <= 1 for v in metrics.values())

    def test_get_training_history(self, sample_data):
        """Test training history retrieval."""
        X, y, feature_names = sample_data

        detector = TransformerDetector(n_features=X.shape[1])
        detector.train(X, y, epochs=3, verbose=0)

        history = detector.get_training_history()

        assert history is not None
        assert 'loss' in history.columns
        assert 'accuracy' in history.columns
        assert len(history) > 0

    def test_save_load(self, sample_data, tmp_path):
        """Test model saving and loading."""
        X, y, feature_names = sample_data

        detector = TransformerDetector(n_features=X.shape[1])
        detector.train(X, y, feature_names=feature_names, epochs=2, verbose=0)

        # Save
        save_path = str(tmp_path / "transformer_test")
        detector.save(save_path)

        # Load
        detector2 = TransformerDetector()
        detector2.load(save_path)

        assert detector2.is_trained
        assert detector2.n_features == detector.n_features
        assert detector2.embed_dim == detector.embed_dim
        assert detector2.feature_names == detector.feature_names

        # Predictions should be similar
        pred1 = detector.predict(X)
        pred2 = detector2.predict(X)
        np.testing.assert_allclose(pred1, pred2, rtol=1e-5)

    def test_untrained_error(self, sample_data):
        """Test error when using untrained model."""
        X, y, feature_names = sample_data

        detector = TransformerDetector(n_features=X.shape[1])

        with pytest.raises(ValueError, match="not trained"):
            detector.predict(X)


class TestDeepLearningModule:
    """Tests for deep learning module utilities."""

    def test_module_import(self):
        """Test that deep learning module can be imported."""
        from src.models import DEEP_LEARNING_AVAILABLE

        if TENSORFLOW_AVAILABLE:
            assert DEEP_LEARNING_AVAILABLE is True
            from src.models import LSTMDetector, TransformerDetector
            assert LSTMDetector is not None
            assert TransformerDetector is not None

    def test_deep_learning_module_factory(self):
        """Test factory function for creating detectors."""
        from src.models.deep_learning import get_detector

        # Test LSTM creation
        lstm = get_detector('lstm', sequence_length=10, n_features=5)
        assert isinstance(lstm, LSTMDetector)

        # Test Transformer creation
        transformer = get_detector('transformer', n_features=5)
        assert isinstance(transformer, TransformerDetector)

        # Test invalid type
        with pytest.raises(ValueError):
            get_detector('invalid_type')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
