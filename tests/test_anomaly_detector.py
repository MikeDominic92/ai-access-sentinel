"""
Tests for anomaly detection module.
"""

import pytest
import numpy as np
import pandas as pd
from src.models.anomaly_detector import AnomalyDetector, EnsembleAnomalyDetector
from src.data.generators import IAMDataGenerator
from src.data.preprocessors import IAMDataPreprocessor


@pytest.fixture
def sample_data():
    """Generate sample IAM data for testing."""
    generator = IAMDataGenerator()
    df = generator.generate_complete_dataset(
        num_users=50,
        normal_events_per_user=20,
        anomaly_ratio=0.1,
        output_path='data/test_sample.csv'
    )
    return df


@pytest.fixture
def preprocessed_data(sample_data):
    """Preprocess sample data."""
    preprocessor = IAMDataPreprocessor()
    df = preprocessor.preprocess_for_training(sample_data)
    feature_cols = preprocessor.get_feature_columns()
    X = df[feature_cols].values
    y = df['is_anomaly'].values if 'is_anomaly' in df.columns else None
    return X, y, feature_cols


def test_anomaly_detector_initialization():
    """Test anomaly detector initialization."""
    detector = AnomalyDetector(algorithm='isolation_forest')
    assert detector.algorithm == 'isolation_forest'
    assert detector.contamination == 0.05
    assert not detector.is_trained


def test_anomaly_detector_training(preprocessed_data):
    """Test anomaly detector training."""
    X, y, feature_cols = preprocessed_data

    detector = AnomalyDetector('isolation_forest')
    detector.train(X, feature_names=feature_cols)

    assert detector.is_trained
    assert detector.model is not None
    assert detector.feature_names == feature_cols


def test_anomaly_detector_prediction(preprocessed_data):
    """Test anomaly detection predictions."""
    X, y, feature_cols = preprocessed_data

    detector = AnomalyDetector('isolation_forest', contamination=0.1)
    detector.train(X, feature_names=feature_cols)

    predictions = detector.predict(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({-1, 1})
    # Check contamination roughly matches
    anomaly_ratio = (predictions == -1).sum() / len(predictions)
    assert 0.05 <= anomaly_ratio <= 0.15  # Allow some variance


def test_anomaly_detector_scoring(preprocessed_data):
    """Test anomaly score generation."""
    X, y, feature_cols = preprocessed_data

    detector = AnomalyDetector('isolation_forest')
    detector.train(X, feature_names=feature_cols)

    scores = detector.score_samples(X)

    assert len(scores) == len(X)
    assert all(isinstance(s, (float, np.floating)) for s in scores)


def test_anomaly_detector_event_analysis(preprocessed_data):
    """Test single event analysis."""
    X, y, feature_cols = preprocessed_data

    detector = AnomalyDetector('isolation_forest')
    detector.train(X, feature_names=feature_cols)

    # Create test event
    features = {col: float(X[0, i]) for i, col in enumerate(feature_cols)}

    result = detector.analyze_event(features)

    assert 'is_anomaly' in result
    assert 'anomaly_score' in result
    assert 'risk_level' in result
    assert 'risk_score' in result
    assert result['risk_level'] in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NORMAL']


def test_different_algorithms(preprocessed_data):
    """Test all anomaly detection algorithms."""
    X, y, feature_cols = preprocessed_data

    algorithms = ['isolation_forest', 'one_class_svm', 'lof']

    for algo in algorithms:
        detector = AnomalyDetector(algo, contamination=0.1)
        detector.train(X, feature_names=feature_cols)

        assert detector.is_trained
        predictions = detector.predict(X)
        assert len(predictions) == len(X)


def test_ensemble_detector(preprocessed_data):
    """Test ensemble anomaly detector."""
    X, y, feature_cols = preprocessed_data

    ensemble = EnsembleAnomalyDetector(contamination=0.1, voting='soft')
    ensemble.train(X, feature_names=feature_cols)

    assert ensemble.is_trained

    predictions = ensemble.predict(X)
    assert len(predictions) == len(X)

    # Test event analysis
    features = {col: float(X[0, i]) for i, col in enumerate(feature_cols)}
    result = ensemble.analyze_event(features)

    assert 'is_anomaly' in result
    assert 'individual_results' in result
    assert 'agreement' in result
    assert len(result['individual_results']) == 3


def test_model_save_load(preprocessed_data, tmp_path):
    """Test model serialization."""
    X, y, feature_cols = preprocessed_data

    # Train and save
    detector = AnomalyDetector('isolation_forest')
    detector.train(X, feature_names=feature_cols)

    model_path = tmp_path / "test_model.joblib"
    detector.save(str(model_path))

    assert model_path.exists()

    # Load and verify
    loaded_detector = AnomalyDetector()
    loaded_detector.load(str(model_path))

    assert loaded_detector.is_trained
    assert loaded_detector.algorithm == 'isolation_forest'
    assert loaded_detector.feature_names == feature_cols

    # Verify predictions match
    original_pred = detector.predict(X[:5])
    loaded_pred = loaded_detector.predict(X[:5])
    np.testing.assert_array_equal(original_pred, loaded_pred)


def test_untrained_model_error():
    """Test that using untrained model raises error."""
    detector = AnomalyDetector()

    with pytest.raises(ValueError, match="Model not trained"):
        detector.predict(np.random.rand(10, 5))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
