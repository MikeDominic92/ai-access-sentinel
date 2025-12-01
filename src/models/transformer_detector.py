"""
Transformer-based anomaly detection for IAM access patterns.

Uses self-attention mechanism to identify important features and patterns.
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


class TransformerBlock(layers.Layer):
    """
    Transformer block with multi-head attention.
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1):
        """
        Initialize transformer block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward network dimension
            dropout_rate: Dropout rate
        """
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        """Forward pass."""
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerDetector:
    """
    Transformer-based anomaly detector for IAM access patterns.

    Good for:
    - Feature importance analysis (which features matter most)
    - Context-aware anomaly detection
    - Handling tabular IAM data with attention mechanism
    - Understanding relationships between features

    Architecture:
    Input (n_features)
    -> Dense embedding (embed_dim)
    -> Transformer Block (multi-head attention)
    -> Global Average Pooling
    -> Dense(32, relu)
    -> Dense(1, sigmoid)
    """

    def __init__(
        self,
        n_features: int = None,
        embed_dim: int = 32,
        num_heads: int = 4,
        ff_dim: int = 64,
        dropout_rate: float = 0.1,
        threshold: float = 0.5,
        random_state: int = 42
    ):
        """
        Initialize Transformer detector.

        Args:
            n_features: Number of input features (set during training if None)
            embed_dim: Embedding dimension (should be divisible by num_heads)
            num_heads: Number of attention heads
            ff_dim: Feed-forward network dimension
            dropout_rate: Dropout rate for regularization
            threshold: Anomaly classification threshold
            random_state: Random seed for reproducibility
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required. Install with: pip install tensorflow>=2.13.0")

        self.n_features = n_features
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.threshold = threshold
        self.random_state = random_state

        self.model = None
        self.attention_model = None  # For visualization
        self.history = None
        self.is_trained = False
        self.feature_names = None

        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

    def _build_model(self) -> keras.Model:
        """
        Build Transformer model architecture.

        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=(self.n_features,), name='input')

        # Reshape to sequence format (batch, 1, features)
        x = layers.Reshape((1, self.n_features))(inputs)

        # Embedding layer
        x = layers.Dense(self.embed_dim, name='embedding')(x)

        # Transformer block
        transformer_block = TransformerBlock(
            self.embed_dim,
            self.num_heads,
            self.ff_dim,
            self.dropout_rate
        )
        x = transformer_block(x)

        # Global pooling
        x = layers.GlobalAveragePooling1D(name='pooling')(x)

        # Dense layers
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(32, activation='relu', name='dense')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name='transformer_anomaly_detector')

        # Compile
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        return model

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1
    ) -> 'TransformerDetector':
        """
        Train Transformer model.

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
        print(f"Training Transformer on {X.shape[0]} samples...")

        # Set n_features if not already set
        if self.n_features is None:
            self.n_features = X.shape[1]

        self.feature_names = feature_names

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
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        # Create attention visualization model
        self._build_attention_model()

        self.is_trained = True
        print("Training complete!")

        return self

    def _build_attention_model(self) -> None:
        """Build model for extracting attention weights."""
        # Get transformer layer
        transformer_layer = None
        for layer in self.model.layers:
            if isinstance(layer, TransformerBlock):
                transformer_layer = layer
                break

        if transformer_layer is not None:
            # Create model that outputs attention weights
            inputs = self.model.input
            # This is a simplified version - full attention extraction would require
            # modifying the TransformerBlock to return attention weights
            self.attention_model = models.Model(
                inputs=inputs,
                outputs=self.model.layers[-2].output  # Dense layer before output
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly probabilities.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Array of anomaly probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        probabilities = self.model.predict(X, verbose=0)
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

    def analyze_event(
        self,
        features: Union[Dict[str, float], np.ndarray]
    ) -> Dict[str, any]:
        """
        Analyze a single access event.

        Args:
            features: Dictionary of feature values or numpy array

        Returns:
            Analysis results with attention information
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Convert to array
        if isinstance(features, dict):
            if self.feature_names:
                X = np.array([features.get(name, 0) for name in self.feature_names]).reshape(1, -1)
            else:
                X = np.array(list(features.values())).reshape(1, -1)
        else:
            X = features.reshape(1, -1)

        # Get prediction
        probability = self.predict(X)[0]
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

        result = {
            'is_anomaly': bool(is_anomaly),
            'anomaly_probability': float(probability),
            'risk_level': risk_level,
            'risk_score': risk_score,
            'detector': 'Transformer'
        }

        # Add feature importance (simplified)
        if self.attention_model is not None and self.feature_names is not None:
            result['top_features'] = self._get_important_features(X)

        return result

    def _get_important_features(
        self,
        X: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, any]]:
        """
        Get most important features for prediction.

        Args:
            X: Feature matrix (1, n_features)
            top_k: Number of top features to return

        Returns:
            List of feature importance dictionaries
        """
        # Simplified feature importance based on feature magnitudes
        # In a full implementation, this would use actual attention weights
        feature_importance = np.abs(X[0])
        top_indices = np.argsort(feature_importance)[-top_k:][::-1]

        important_features = []
        for idx in top_indices:
            important_features.append({
                'feature': self.feature_names[idx] if self.feature_names else f'f{idx}',
                'value': float(X[0, idx]),
                'importance': float(feature_importance[idx])
            })

        return important_features

    def get_attention_weights(
        self,
        X: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Get attention weights for samples.

        Note: This is a simplified version. Full implementation would require
        modifying TransformerBlock to return attention weights.

        Args:
            X: Feature matrix

        Returns:
            Attention weights (if available)
        """
        if not self.is_trained or self.attention_model is None:
            return None

        # Get intermediate representation
        representation = self.attention_model.predict(X, verbose=0)
        return representation

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

        # Evaluate
        results = self.model.evaluate(X, y, verbose=0)

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
        model_path = f"{path}/transformer_model"
        self.model.save(model_path)

        # Save configuration
        config = {
            'n_features': self.n_features,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            'threshold': self.threshold,
            'random_state': self.random_state,
            'feature_names': self.feature_names
        }

        import json
        with open(f"{path}/transformer_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {path}")

    def load(self, path: str) -> 'TransformerDetector':
        """
        Load model from disk.

        Args:
            path: Directory path to load model from

        Returns:
            self for method chaining
        """
        import json

        # Load configuration
        with open(f"{path}/transformer_config.json", 'r') as f:
            config = json.load(f)

        self.n_features = config['n_features']
        self.embed_dim = config['embed_dim']
        self.num_heads = config['num_heads']
        self.ff_dim = config['ff_dim']
        self.dropout_rate = config['dropout_rate']
        self.threshold = config['threshold']
        self.random_state = config['random_state']
        self.feature_names = config['feature_names']

        # Load model
        model_path = f"{path}/transformer_model"
        self.model = keras.models.load_model(
            model_path,
            custom_objects={'TransformerBlock': TransformerBlock}
        )

        self._build_attention_model()

        self.is_trained = True
        print(f"Model loaded from {path}")

        return self


if __name__ == "__main__":
    # Example usage
    print("Transformer Detector Example")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 15

    # Normal samples
    X_normal = np.random.randn(800, n_features) * 0.5
    y_normal = np.zeros(800)

    # Anomalous samples (different distribution)
    X_anomaly = np.random.randn(200, n_features) * 2.0 + 2.5
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

    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]

    # Train Transformer
    detector = TransformerDetector(
        n_features=n_features,
        embed_dim=32,
        num_heads=4,
        ff_dim=64
    )
    detector.train(X, y, feature_names=feature_names, epochs=20, verbose=1)

    # Evaluate
    metrics = detector.evaluate(X, y)
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Test prediction
    test_sample = X[0]
    result = detector.analyze_event(test_sample)
    print("\nSample Event Analysis:")
    print(f"  Anomaly: {result['is_anomaly']}")
    print(f"  Probability: {result['anomaly_probability']:.4f}")
    print(f"  Risk Level: {result['risk_level']}")
    if 'top_features' in result:
        print(f"  Top Features:")
        for feat in result['top_features'][:3]:
            print(f"    - {feat['feature']}: {feat['value']:.3f}")
