"""
LSTM-based anomaly detection for access sequences.

Detects temporal patterns and multi-step attack patterns in IAM access logs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not available. Install with: pip install tensorflow>=2.13.0")


class LSTMDetector:
    """
    LSTM-based anomaly detector for sequential access patterns.

    Good for:
    - Temporal pattern analysis (order of access matters)
    - Multi-step attack detection (reconnaissance -> access -> exfiltration)
    - Time-series behavioral analysis
    - Detecting gradual privilege escalation

    Architecture:
    Input (sequence_length, n_features)
    -> LSTM(64)
    -> Dropout(0.2)
    -> LSTM(32)
    -> Dropout(0.2)
    -> Dense(16, relu)
    -> Dense(1, sigmoid)
    """

    def __init__(
        self,
        sequence_length: int = 10,
        n_features: int = None,
        lstm_units: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        threshold: float = 0.5,
        random_state: int = 42
    ):
        """
        Initialize LSTM detector.

        Args:
            sequence_length: Number of events in each sequence
            n_features: Number of features per event (set during training if None)
            lstm_units: Number of units in each LSTM layer
            dropout_rate: Dropout rate for regularization
            threshold: Anomaly classification threshold
            random_state: Random seed for reproducibility
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required. Install with: pip install tensorflow>=2.13.0")

        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.threshold = threshold
        self.random_state = random_state

        self.model = None
        self.history = None
        self.is_trained = False
        self.feature_names = None

        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

    def _build_model(self) -> keras.Model:
        """
        Build LSTM model architecture.

        Returns:
            Compiled Keras model
        """
        model = models.Sequential(name='lstm_anomaly_detector')

        # First LSTM layer with return sequences
        model.add(layers.LSTM(
            self.lstm_units[0],
            input_shape=(self.sequence_length, self.n_features),
            return_sequences=True if len(self.lstm_units) > 1 else False,
            name='lstm_1'
        ))
        model.add(layers.Dropout(self.dropout_rate, name='dropout_1'))

        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:], start=2):
            return_seq = i < len(self.lstm_units)
            model.add(layers.LSTM(
                units,
                return_sequences=return_seq,
                name=f'lstm_{i}'
            ))
            model.add(layers.Dropout(self.dropout_rate, name=f'dropout_{i}'))

        # Dense layers
        model.add(layers.Dense(16, activation='relu', name='dense_1'))
        model.add(layers.Dense(1, activation='sigmoid', name='output'))

        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        return model

    def _create_sequences(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Convert data into sequences for LSTM.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (optional)

        Returns:
            Sequences (and labels if y provided)
        """
        n_samples = len(X)
        n_sequences = n_samples - self.sequence_length + 1

        # Create sequences
        X_sequences = np.zeros((n_sequences, self.sequence_length, self.n_features))
        for i in range(n_sequences):
            X_sequences[i] = X[i:i + self.sequence_length]

        if y is not None:
            # Use label from last event in sequence
            y_sequences = y[self.sequence_length - 1:]
            return X_sequences, y_sequences

        return X_sequences

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1
    ) -> 'LSTMDetector':
        """
        Train LSTM model on sequential access data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (0=normal, 1=anomaly)
            feature_names: Names of features
            validation_split: Fraction of data for validation
            epochs: Maximum training epochs
            batch_size: Batch size for training
            verbose: Verbosity level

        Returns:
            self for method chaining
        """
        print(f"Training LSTM on {X.shape[0]} samples...")
        print(f"Creating sequences of length {self.sequence_length}...")

        # Set n_features if not already set
        if self.n_features is None:
            self.n_features = X.shape[1]

        self.feature_names = feature_names

        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y)
        print(f"Created {X_seq.shape[0]} sequences")

        # Build model
        self.model = self._build_model()

        if verbose:
            self.model.summary()

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=verbose
            )
        ]

        # Train model
        self.history = self.model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        self.is_trained = True
        print("Training complete!")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly probabilities for sequences.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Array of anomaly probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Create sequences
        X_seq = self._create_sequences(X)

        # Predict
        probabilities = self.model.predict(X_seq, verbose=0)

        return probabilities.flatten()

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly classes (0 or 1).

        Args:
            X: Feature matrix

        Returns:
            Array of predictions (1=anomaly, 0=normal)
        """
        probabilities = self.predict(X)
        return (probabilities >= self.threshold).astype(int)

    def analyze_sequence(
        self,
        sequence: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Analyze a single access sequence.

        Args:
            sequence: Sequence of events (sequence_length, n_features)
            feature_names: Feature names for interpretability

        Returns:
            Analysis results
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        if sequence.shape[0] != self.sequence_length:
            raise ValueError(f"Sequence must have {self.sequence_length} events")

        # Reshape for prediction
        X = sequence.reshape(1, self.sequence_length, self.n_features)

        # Get prediction
        probability = self.model.predict(X, verbose=0)[0][0]
        is_anomaly = probability >= self.threshold

        # Determine risk level
        if probability >= 0.9:
            risk_level = "CRITICAL"
            risk_score = 95
        elif probability >= 0.7:
            risk_level = "HIGH"
            risk_score = 85
        elif probability >= 0.5:
            risk_level = "MEDIUM"
            risk_score = 70
        elif probability >= 0.3:
            risk_level = "LOW"
            risk_score = 55
        else:
            risk_level = "NORMAL"
            risk_score = 30

        return {
            'is_anomaly': bool(is_anomaly),
            'anomaly_probability': float(probability),
            'risk_level': risk_level,
            'risk_score': risk_score,
            'sequence_length': self.sequence_length,
            'detector': 'LSTM'
        }

    def detect_attack_patterns(
        self,
        X: np.ndarray,
        user_ids: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Detect multi-step attack patterns in access sequences.

        Args:
            X: Feature matrix
            user_ids: User IDs for each event (optional)

        Returns:
            DataFrame with detected attack patterns
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Get predictions
        probabilities = self.predict(X)

        # Find high-risk sequences
        attack_indices = np.where(probabilities >= self.threshold)[0]

        results = []
        for idx in attack_indices:
            # Sequence spans from idx to idx + sequence_length - 1
            start_idx = idx
            end_idx = idx + self.sequence_length - 1

            result = {
                'sequence_start': start_idx,
                'sequence_end': end_idx,
                'anomaly_probability': probabilities[idx],
                'pattern_type': self._classify_pattern(probabilities[idx])
            }

            if user_ids is not None and end_idx < len(user_ids):
                result['user_id'] = user_ids[end_idx]

            results.append(result)

        return pd.DataFrame(results)

    def _classify_pattern(self, probability: float) -> str:
        """Classify attack pattern based on probability."""
        if probability >= 0.9:
            return "Multi-step attack (likely exfiltration)"
        elif probability >= 0.7:
            return "Privilege escalation pattern"
        elif probability >= 0.5:
            return "Reconnaissance pattern"
        else:
            return "Suspicious sequence"

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Dictionary of metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y)

        # Evaluate
        results = self.model.evaluate(X_seq, y_seq, verbose=0)

        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3]
        }

        # Calculate F1 score
        if metrics['precision'] > 0 and metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                                  (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0

        return metrics

    def get_training_history(self) -> Optional[pd.DataFrame]:
        """
        Get training history as DataFrame.

        Returns:
            DataFrame with training metrics or None
        """
        if self.history is None:
            return None

        return pd.DataFrame(self.history.history)

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Directory path to save model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        # Save model
        model_path = f"{path}/lstm_model"
        self.model.save(model_path)

        # Save configuration
        config = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'threshold': self.threshold,
            'random_state': self.random_state,
            'feature_names': self.feature_names
        }

        import json
        with open(f"{path}/lstm_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {path}")

    def load(self, path: str) -> 'LSTMDetector':
        """
        Load model from disk.

        Args:
            path: Directory path to load model from

        Returns:
            self for method chaining
        """
        import json

        # Load configuration
        with open(f"{path}/lstm_config.json", 'r') as f:
            config = json.load(f)

        self.sequence_length = config['sequence_length']
        self.n_features = config['n_features']
        self.lstm_units = config['lstm_units']
        self.dropout_rate = config['dropout_rate']
        self.threshold = config['threshold']
        self.random_state = config['random_state']
        self.feature_names = config['feature_names']

        # Load model
        model_path = f"{path}/lstm_model"
        self.model = keras.models.load_model(model_path)

        self.is_trained = True
        print(f"Model loaded from {path}")

        return self


if __name__ == "__main__":
    # Example usage
    print("LSTM Detector Example")
    print("=" * 60)

    # Generate synthetic sequence data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Normal sequences
    X_normal = np.random.randn(800, n_features) * 0.5
    y_normal = np.zeros(800)

    # Anomalous sequences (different distribution)
    X_anomaly = np.random.randn(200, n_features) * 2.0 + 3.0
    y_anomaly = np.ones(200)

    # Combine
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([y_normal, y_anomaly])

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Anomaly rate: {y.mean():.2%}")

    # Train LSTM
    detector = LSTMDetector(sequence_length=10, n_features=n_features)
    detector.train(X, y, epochs=20, verbose=1)

    # Evaluate
    metrics = detector.evaluate(X, y)
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Test prediction
    test_sequence = X[:10]
    result = detector.analyze_sequence(test_sequence)
    print("\nSample Sequence Analysis:")
    print(f"  Anomaly: {result['is_anomaly']}")
    print(f"  Probability: {result['anomaly_probability']:.4f}")
    print(f"  Risk Level: {result['risk_level']}")
