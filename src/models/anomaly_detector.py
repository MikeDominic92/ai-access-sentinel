"""
Anomaly detection for IAM access patterns.

Implements multiple algorithms: Isolation Forest, One-Class SVM, Local Outlier Factor.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """
    Multi-algorithm anomaly detection for IAM access patterns.

    Supports:
    - Isolation Forest: Fast, works well with high-dimensional data
    - One-Class SVM: Effective for complex boundaries
    - Local Outlier Factor: Good for density-based anomalies
    """

    def __init__(
        self,
        algorithm: str = 'isolation_forest',
        contamination: float = 0.05,
        random_state: int = 42
    ):
        """
        Initialize anomaly detector.

        Args:
            algorithm: 'isolation_forest', 'one_class_svm', or 'lof'
            contamination: Expected proportion of anomalies (0.01 to 0.5)
            random_state: Random seed for reproducibility
        """
        self.algorithm = algorithm
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False

    def _create_model(self):
        """Create the appropriate model based on algorithm."""
        if self.algorithm == 'isolation_forest':
            return IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=100,
                max_samples='auto',
                n_jobs=-1
            )
        elif self.algorithm == 'one_class_svm':
            return OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='auto'
            )
        elif self.algorithm == 'lof':
            return LocalOutlierFactor(
                contamination=self.contamination,
                n_neighbors=20,
                novelty=True,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def train(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'AnomalyDetector':
        """
        Train the anomaly detection model.

        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Names of features for interpretability

        Returns:
            self for method chaining
        """
        print(f"Training {self.algorithm} on {X.shape[0]} samples...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_scaled)

        self.feature_names = feature_names
        self.is_trained = True

        print(f"Training complete!")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (1 = normal, -1 = anomaly).

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (lower = more anomalous).

        Args:
            X: Feature matrix

        Returns:
            Array of anomaly scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        X_scaled = self.scaler.transform(X)

        if hasattr(self.model, 'score_samples'):
            scores = self.model.score_samples(X_scaled)
        else:
            # For LOF in novelty mode
            scores = self.model.decision_function(X_scaled)

        return scores

    def analyze_event(
        self,
        features: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Analyze a single access event.

        Args:
            features: Dictionary of feature values

        Returns:
            Analysis results with anomaly flag and score
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Convert features to array
        if self.feature_names:
            X = np.array([features.get(name, 0) for name in self.feature_names]).reshape(1, -1)
        else:
            X = np.array(list(features.values())).reshape(1, -1)

        # Get prediction and score
        prediction = self.predict(X)[0]
        score = self.score_samples(X)[0]

        # Determine risk level
        if score < -0.5:
            risk_level = "CRITICAL"
            risk_score = 95
        elif score < -0.3:
            risk_level = "HIGH"
            risk_score = 85
        elif score < -0.1:
            risk_level = "MEDIUM"
            risk_score = 70
        elif score < 0:
            risk_level = "LOW"
            risk_score = 55
        else:
            risk_level = "NORMAL"
            risk_score = 30

        return {
            'is_anomaly': prediction == -1,
            'anomaly_score': float(score),
            'risk_level': risk_level,
            'risk_score': risk_score,
            'algorithm': self.algorithm
        }

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance (only for Isolation Forest).

        Returns:
            DataFrame with feature importances or None
        """
        if self.algorithm != 'isolation_forest' or not self.is_trained:
            return None

        if not hasattr(self.model, 'feature_importances_'):
            return None

        importances = pd.DataFrame({
            'feature': self.feature_names or [f'f{i}' for i in range(len(self.model.feature_importances_))],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importances

    def train_from_file(
        self,
        file_path: str,
        feature_columns: Optional[List[str]] = None
    ) -> 'AnomalyDetector':
        """
        Train model from CSV file.

        Args:
            file_path: Path to CSV file
            feature_columns: Columns to use as features

        Returns:
            self for method chaining
        """
        from src.data.preprocessors import IAMDataPreprocessor

        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)

        # Preprocess
        preprocessor = IAMDataPreprocessor()
        df = preprocessor.preprocess_for_training(df)

        # Get features
        if feature_columns is None:
            feature_columns = preprocessor.get_feature_columns()

        X = df[feature_columns].values

        # Train
        self.train(X, feature_names=feature_columns)

        return self

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: File path to save model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'algorithm': self.algorithm,
            'contamination': self.contamination,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }, path)

        print(f"Model saved to {path}")

    def load(self, path: str) -> 'AnomalyDetector':
        """
        Load model from disk.

        Args:
            path: File path to load model from

        Returns:
            self for method chaining
        """
        data = joblib.load(path)

        self.model = data['model']
        self.scaler = data['scaler']
        self.algorithm = data['algorithm']
        self.contamination = data['contamination']
        self.feature_names = data['feature_names']
        self.is_trained = data['is_trained']

        print(f"Model loaded from {path}")
        return self


class EnsembleAnomalyDetector:
    """
    Ensemble of multiple anomaly detectors for robust detection.

    Combines Isolation Forest, One-Class SVM, and LOF.
    """

    def __init__(self, contamination: float = 0.05, voting: str = 'soft'):
        """
        Initialize ensemble detector.

        Args:
            contamination: Expected proportion of anomalies
            voting: 'hard' (majority vote) or 'soft' (average scores)
        """
        self.contamination = contamination
        self.voting = voting

        self.detectors = {
            'isolation_forest': AnomalyDetector('isolation_forest', contamination),
            'one_class_svm': AnomalyDetector('one_class_svm', contamination),
            'lof': AnomalyDetector('lof', contamination)
        }

        self.is_trained = False

    def train(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """Train all detectors in the ensemble."""
        print("Training ensemble of detectors...")

        for name, detector in self.detectors.items():
            detector.train(X, feature_names)

        self.is_trained = True
        print("Ensemble training complete!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction."""
        if not self.is_trained:
            raise ValueError("Ensemble not trained")

        if self.voting == 'hard':
            # Majority vote
            predictions = []
            for detector in self.detectors.values():
                predictions.append(detector.predict(X))

            predictions = np.array(predictions)
            # Return -1 if majority say anomaly
            ensemble_pred = np.where(
                np.sum(predictions == -1, axis=0) >= 2, -1, 1
            )

        else:  # soft voting
            # Average scores
            scores = []
            for detector in self.detectors.values():
                scores.append(detector.score_samples(X))

            avg_scores = np.mean(scores, axis=0)
            threshold = np.percentile(avg_scores, self.contamination * 100)
            ensemble_pred = np.where(avg_scores < threshold, -1, 1)

        return ensemble_pred

    def analyze_event(self, features: Dict[str, float]) -> Dict[str, any]:
        """Analyze event with ensemble."""
        results = {}

        for name, detector in self.detectors.items():
            results[name] = detector.analyze_event(features)

        # Ensemble decision
        anomaly_votes = sum(1 for r in results.values() if r['is_anomaly'])
        avg_score = np.mean([r['anomaly_score'] for r in results.values()])
        avg_risk = np.mean([r['risk_score'] for r in results.values()])

        return {
            'is_anomaly': anomaly_votes >= 2,
            'anomaly_score': float(avg_score),
            'risk_score': int(avg_risk),
            'individual_results': results,
            'agreement': anomaly_votes
        }


if __name__ == "__main__":
    # Example usage
    from src.data.generators import IAMDataGenerator
    from src.data.preprocessors import IAMDataPreprocessor

    # Generate data
    print("Generating training data...")
    generator = IAMDataGenerator()
    df = generator.generate_complete_dataset(
        num_users=100,
        normal_events_per_user=50,
        output_path='data/sample_iam_logs.csv'
    )

    # Preprocess
    print("\nPreprocessing...")
    preprocessor = IAMDataPreprocessor()
    df = preprocessor.preprocess_for_training(df)

    # Prepare features
    feature_cols = preprocessor.get_feature_columns()
    X = df[feature_cols].values
    y_true = df['is_anomaly'].values if 'is_anomaly' in df.columns else None

    # Train Isolation Forest
    print("\n" + "="*50)
    print("Testing Isolation Forest")
    print("="*50)
    detector_if = AnomalyDetector('isolation_forest', contamination=0.05)
    detector_if.train(X, feature_names=feature_cols)

    # Evaluate
    predictions = detector_if.predict(X)
    if y_true is not None:
        from sklearn.metrics import classification_report
        print("\nClassification Report:")
        print(classification_report(
            y_true,
            predictions == -1,
            target_names=['Normal', 'Anomaly']
        ))

    # Save model
    detector_if.save('models/trained/anomaly_detector_if.joblib')

    # Test ensemble
    print("\n" + "="*50)
    print("Testing Ensemble Detector")
    print("="*50)
    ensemble = EnsembleAnomalyDetector(contamination=0.05)
    ensemble.train(X, feature_names=feature_cols)

    # Example event analysis
    print("\n" + "="*50)
    print("Analyzing Sample Event")
    print("="*50)

    sample_features = {col: X[0, i] for i, col in enumerate(feature_cols)}
    result = ensemble.analyze_event(sample_features)

    print(f"\nAnalysis Result:")
    print(f"  Is Anomaly: {result['is_anomaly']}")
    print(f"  Risk Score: {result['risk_score']}")
    print(f"  Agreement: {result['agreement']}/3 detectors")
