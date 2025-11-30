"""
Access request prediction model.

Predicts whether access requests should be approved based on peer analysis.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from typing import Dict, List, Optional, Tuple


class AccessPredictor:
    """
    Predict access approval using Random Forest.

    Analyzes:
    - User department and role
    - Resource sensitivity
    - Peer access patterns
    - Historical approval rates
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize access predictor.

        Args:
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names = None
        self.is_trained = False
        self.random_state = random_state

    def prepare_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare features for training or prediction.

        Args:
            df: DataFrame with access data
            fit: Whether to fit encoders

        Returns:
            Tuple of (X, y) where y is None for prediction
        """
        df = df.copy()

        # Categorical features to encode
        categorical_cols = ['department', 'job_title', 'resource', 'action']

        # Encode categorical features
        for col in categorical_cols:
            if col not in df.columns:
                continue

            if fit:
                encoder = LabelEncoder()
                df[f'{col}_encoded'] = encoder.fit_transform(df[col].astype(str))
                self.label_encoders[col] = encoder
            else:
                if col in self.label_encoders:
                    encoder = self.label_encoders[col]
                    df[f'{col}_encoded'] = df[col].astype(str).apply(
                        lambda x: encoder.transform([x])[0]
                        if x in encoder.classes_
                        else -1
                    )
                else:
                    df[f'{col}_encoded'] = -1

        # Numerical features
        feature_cols = [
            'department_encoded', 'job_title_encoded',
            'resource_encoded', 'action_encoded'
        ]

        # Add additional features if available
        optional_features = [
            'resource_sensitivity_score', 'action_risk_score',
            'combined_risk_score', 'user_total_events',
            'user_unique_resources', 'user_success_rate',
            'is_business_hours', 'is_suspicious_location'
        ]

        for col in optional_features:
            if col in df.columns:
                feature_cols.append(col)

        self.feature_names = feature_cols
        X = df[feature_cols].values

        # Scale features
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        # Get labels if available
        y = df['success'].values if 'success' in df.columns else None

        return X, y

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the access prediction model.

        Args:
            df: DataFrame with historical access data
            test_size: Proportion of data for testing

        Returns:
            Dictionary with performance metrics
        """
        print(f"Training access predictor on {len(df)} samples...")

        # Prepare features
        X, y = self.prepare_features(df, fit=True)

        if y is None:
            raise ValueError("DataFrame must have 'success' column for training")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_test)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Denied', 'Approved']))

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }

        print(f"\nOverall Accuracy: {metrics['accuracy']:.3f}")

        return metrics

    def predict_access(
        self,
        user_info: Dict[str, any],
        resource: str,
        action: str
    ) -> Dict[str, any]:
        """
        Predict whether access should be approved.

        Args:
            user_info: Dictionary with user information
            resource: Resource being accessed
            action: Action being performed

        Returns:
            Prediction result with confidence
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Create DataFrame for prediction
        pred_df = pd.DataFrame([{
            'department': user_info.get('department', 'Unknown'),
            'job_title': user_info.get('job_title', 'Unknown'),
            'resource': resource,
            'action': action,
            'resource_sensitivity_score': user_info.get('resource_sensitivity_score', 2),
            'action_risk_score': user_info.get('action_risk_score', 2),
            'combined_risk_score': user_info.get('combined_risk_score', 4),
            'user_total_events': user_info.get('user_total_events', 0),
            'user_unique_resources': user_info.get('user_unique_resources', 0),
            'user_success_rate': user_info.get('user_success_rate', 1.0),
            'is_business_hours': user_info.get('is_business_hours', 1),
            'is_suspicious_location': user_info.get('is_suspicious_location', 0)
        }])

        # Prepare features
        X, _ = self.prepare_features(pred_df, fit=False)

        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        # Get confidence
        confidence = probabilities[1] if prediction == 1 else probabilities[0]

        # Recommendation
        if prediction == 1 and confidence > 0.8:
            recommendation = "APPROVE"
        elif prediction == 1 and confidence > 0.6:
            recommendation = "APPROVE_WITH_REVIEW"
        elif prediction == 0 and confidence > 0.8:
            recommendation = "DENY"
        else:
            recommendation = "MANUAL_REVIEW"

        return {
            'should_approve': bool(prediction),
            'confidence': float(confidence),
            'probability_approve': float(probabilities[1]),
            'probability_deny': float(probabilities[0]),
            'recommendation': recommendation
        }

    def analyze_peer_access(
        self,
        df: pd.DataFrame,
        user_department: str,
        user_job_title: str,
        resource: str
    ) -> Dict[str, any]:
        """
        Analyze how many peers have access to a resource.

        Args:
            df: Historical access data
            user_department: User's department
            user_job_title: User's job title
            resource: Resource to check

        Returns:
            Peer analysis results
        """
        # Find peers (same department or similar job title)
        peers = df[
            ((df['department'] == user_department) |
             (df['job_title'] == user_job_title)) &
            (df['success'] == True)
        ]

        if len(peers) == 0:
            return {
                'peer_count': 0,
                'peers_with_access': 0,
                'peer_access_ratio': 0.0,
                'common_actions': []
            }

        # Check resource access
        peers_with_resource = peers[peers['resource'] == resource]

        total_peers = peers['user_id'].nunique()
        peers_with_access = peers_with_resource['user_id'].nunique()
        access_ratio = peers_with_access / total_peers if total_peers > 0 else 0

        # Common actions on this resource
        if len(peers_with_resource) > 0:
            common_actions = peers_with_resource['action'].value_counts().head(3).to_dict()
        else:
            common_actions = []

        return {
            'peer_count': total_peers,
            'peers_with_access': peers_with_access,
            'peer_access_ratio': access_ratio,
            'common_actions': common_actions
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the Random Forest.

        Returns:
            DataFrame with feature importances
        """
        if not self.is_trained:
            raise ValueError("Model not trained")

        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importances

    def save(self, path: str) -> None:
        """Save model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }, path)

        print(f"Model saved to {path}")

    def load(self, path: str) -> 'AccessPredictor':
        """Load model from disk."""
        data = joblib.load(path)

        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_names = data['feature_names']
        self.is_trained = data['is_trained']

        print(f"Model loaded from {path}")
        return self


if __name__ == "__main__":
    # Example usage
    from src.data.generators import IAMDataGenerator
    from src.data.preprocessors import IAMDataPreprocessor

    # Generate data
    print("Generating training data...")
    generator = IAMDataGenerator()
    df = generator.generate_complete_dataset(
        num_users=150,
        normal_events_per_user=60,
        output_path='data/sample_iam_logs.csv'
    )

    # Preprocess
    print("\nPreprocessing...")
    preprocessor = IAMDataPreprocessor()
    df = preprocessor.preprocess_for_training(df)

    # Train predictor
    print("\n" + "="*50)
    print("Training Access Predictor")
    print("="*50)
    predictor = AccessPredictor()
    metrics = predictor.train(df)

    # Feature importance
    print("\nTop 10 Important Features:")
    print(predictor.get_feature_importance().head(10))

    # Save model
    predictor.save('models/trained/access_predictor.joblib')

    # Test prediction
    print("\n" + "="*50)
    print("Testing Access Prediction")
    print("="*50)

    test_user = {
        'department': 'Engineering',
        'job_title': 'Software Engineer',
        'resource_sensitivity_score': 3,
        'action_risk_score': 2,
        'combined_risk_score': 6,
        'user_total_events': 120,
        'user_unique_resources': 8,
        'user_success_rate': 0.95,
        'is_business_hours': 1,
        'is_suspicious_location': 0
    }

    result = predictor.predict_access(
        test_user,
        resource='source_code_repo',
        action='read'
    )

    print(f"\nAccess Request Analysis:")
    print(f"  Resource: source_code_repo")
    print(f"  Action: read")
    print(f"  Should Approve: {result['should_approve']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Recommendation: {result['recommendation']}")

    # Peer analysis
    print("\n" + "="*50)
    print("Peer Access Analysis")
    print("="*50)

    peer_analysis = predictor.analyze_peer_access(
        df,
        user_department='Engineering',
        user_job_title='Software Engineer',
        resource='source_code_repo'
    )

    print(f"\nPeer Analysis:")
    print(f"  Total Peers: {peer_analysis['peer_count']}")
    print(f"  Peers with Access: {peer_analysis['peers_with_access']}")
    print(f"  Access Ratio: {peer_analysis['peer_access_ratio']:.2%}")
