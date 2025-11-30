"""
Feature engineering for IAM access patterns.

Creates ML-ready features from raw IAM logs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler


class IAMFeatureExtractor:
    """Extract and engineer features from IAM logs."""

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None

    def extract_sequence_features(
        self,
        df: pd.DataFrame,
        user_id: str,
        window_hours: int = 24
    ) -> Dict[str, float]:
        """
        Extract sequence-based features for a user within time window.

        Args:
            df: IAM logs DataFrame
            user_id: User to analyze
            window_hours: Time window in hours

        Returns:
            Dictionary of sequence features
        """
        # Filter to user and recent time window
        cutoff = df['timestamp'].max() - timedelta(hours=window_hours)
        user_events = df[
            (df['user_id'] == user_id) &
            (df['timestamp'] >= cutoff)
        ].sort_values('timestamp')

        if len(user_events) == 0:
            return {
                'event_count_24h': 0,
                'unique_resources_24h': 0,
                'unique_actions_24h': 0,
                'avg_time_between_events': 0,
                'failed_attempts_24h': 0
            }

        # Count features
        event_count = len(user_events)
        unique_resources = user_events['resource'].nunique()
        unique_actions = user_events['action'].nunique()
        failed_attempts = (~user_events['success']).sum()

        # Time between events
        if len(user_events) > 1:
            time_diffs = user_events['timestamp'].diff().dt.total_seconds().dropna()
            avg_time_between = time_diffs.mean()
        else:
            avg_time_between = 0

        return {
            'event_count_24h': event_count,
            'unique_resources_24h': unique_resources,
            'unique_actions_24h': unique_actions,
            'avg_time_between_events': avg_time_between,
            'failed_attempts_24h': failed_attempts
        }

    def extract_peer_comparison_features(
        self,
        df: pd.DataFrame,
        user_id: str,
        department: str
    ) -> Dict[str, float]:
        """
        Compare user behavior to departmental peers.

        Args:
            df: IAM logs DataFrame
            user_id: User to analyze
            department: User's department

        Returns:
            Dictionary of peer comparison features
        """
        # Get user's metrics
        user_data = df[df['user_id'] == user_id]
        user_resource_count = user_data['resource'].nunique()
        user_event_count = len(user_data)

        # Get peer metrics (same department, different user)
        peer_data = df[
            (df['department'] == department) &
            (df['user_id'] != user_id)
        ]

        if len(peer_data) == 0:
            return {
                'resource_count_vs_peers': 0,
                'event_count_vs_peers': 0,
                'peer_deviation_score': 0
            }

        # Peer statistics
        peer_resource_counts = peer_data.groupby('user_id')['resource'].nunique()
        peer_event_counts = peer_data.groupby('user_id').size()

        peer_avg_resources = peer_resource_counts.mean()
        peer_avg_events = peer_event_counts.mean()
        peer_std_resources = peer_resource_counts.std()
        peer_std_events = peer_event_counts.std()

        # Calculate deviations
        resource_deviation = (
            (user_resource_count - peer_avg_resources) / (peer_std_resources + 1)
        )
        event_deviation = (
            (user_event_count - peer_avg_events) / (peer_std_events + 1)
        )

        # Combined deviation score
        deviation_score = np.sqrt(resource_deviation**2 + event_deviation**2)

        return {
            'resource_count_vs_peers': resource_deviation,
            'event_count_vs_peers': event_deviation,
            'peer_deviation_score': deviation_score
        }

    def extract_access_pattern_features(
        self,
        df: pd.DataFrame,
        user_id: str
    ) -> Dict[str, float]:
        """
        Extract access pattern features for a user.

        Args:
            df: IAM logs DataFrame
            user_id: User to analyze

        Returns:
            Dictionary of access pattern features
        """
        user_data = df[df['user_id'] == user_id]

        if len(user_data) == 0:
            return {
                'access_diversity': 0,
                'time_diversity': 0,
                'location_diversity': 0,
                'high_risk_ratio': 0
            }

        # Resource diversity (entropy)
        resource_counts = user_data['resource'].value_counts(normalize=True)
        access_diversity = -np.sum(resource_counts * np.log(resource_counts + 1e-10))

        # Time diversity
        hour_counts = user_data['hour'].value_counts(normalize=True)
        time_diversity = -np.sum(hour_counts * np.log(hour_counts + 1e-10))

        # Location diversity
        location_counts = user_data['location'].value_counts(normalize=True)
        location_diversity = -np.sum(location_counts * np.log(location_counts + 1e-10))

        # High risk action ratio
        if 'combined_risk_score' in user_data.columns:
            high_risk_ratio = (user_data['combined_risk_score'] >= 6).mean()
        else:
            high_risk_ratio = 0

        return {
            'access_diversity': access_diversity,
            'time_diversity': time_diversity,
            'location_diversity': location_diversity,
            'high_risk_ratio': high_risk_ratio
        }

    def create_feature_matrix(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str] = None,
        numerical_cols: List[str] = None,
        fit_encoders: bool = True
    ) -> np.ndarray:
        """
        Create feature matrix for ML models.

        Args:
            df: Preprocessed DataFrame
            categorical_cols: Categorical columns to encode
            numerical_cols: Numerical columns to use
            fit_encoders: Whether to fit new encoders

        Returns:
            Feature matrix as numpy array
        """
        if categorical_cols is None:
            categorical_cols = ['action', 'resource_sensitivity', 'time_category']

        if numerical_cols is None:
            numerical_cols = [
                'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
                'user_total_events', 'user_unique_resources', 'user_success_rate',
                'resource_sensitivity_score', 'action_risk_score',
                'combined_risk_score', 'is_suspicious_location',
                'is_different_location'
            ]

        # Encode categorical variables
        encoded_features = []

        for col in categorical_cols:
            if col not in df.columns:
                continue

            if fit_encoders or col not in self.label_encoders:
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(df[col].astype(str))
                self.label_encoders[col] = encoder
            else:
                encoder = self.label_encoders[col]
                # Handle unseen categories
                encoded = df[col].astype(str).apply(
                    lambda x: encoder.transform([x])[0]
                    if x in encoder.classes_
                    else -1
                )

            encoded_features.append(encoded.reshape(-1, 1))

        # Add numerical features
        numerical_features = df[
            [col for col in numerical_cols if col in df.columns]
        ].values

        # Combine all features
        if encoded_features:
            X = np.hstack([numerical_features] + encoded_features)
        else:
            X = numerical_features

        # Scale features
        if fit_encoders:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        elif self.scaler is not None:
            X = self.scaler.transform(X)

        return X

    def extract_all_features(
        self,
        df: pd.DataFrame,
        include_advanced: bool = True
    ) -> pd.DataFrame:
        """
        Extract all features from IAM logs.

        Args:
            df: Preprocessed DataFrame
            include_advanced: Whether to include sequence and peer features

        Returns:
            DataFrame with all features
        """
        result_df = df.copy()

        if not include_advanced:
            return result_df

        # Add sequence features for each user
        sequence_features_list = []
        for user_id in df['user_id'].unique():
            seq_features = self.extract_sequence_features(df, user_id)
            seq_features['user_id'] = user_id
            sequence_features_list.append(seq_features)

        seq_df = pd.DataFrame(sequence_features_list)
        result_df = result_df.merge(seq_df, on='user_id', how='left')

        # Add peer comparison features
        if 'department' in df.columns:
            peer_features_list = []
            for user_id in df['user_id'].unique():
                user_dept = df[df['user_id'] == user_id]['department'].iloc[0]
                peer_features = self.extract_peer_comparison_features(
                    df, user_id, user_dept
                )
                peer_features['user_id'] = user_id
                peer_features_list.append(peer_features)

            peer_df = pd.DataFrame(peer_features_list)
            result_df = result_df.merge(peer_df, on='user_id', how='left')

        # Add access pattern features
        pattern_features_list = []
        for user_id in df['user_id'].unique():
            pattern_features = self.extract_access_pattern_features(df, user_id)
            pattern_features['user_id'] = user_id
            pattern_features_list.append(pattern_features)

        pattern_df = pd.DataFrame(pattern_features_list)
        result_df = result_df.merge(pattern_df, on='user_id', how='left')

        return result_df


if __name__ == "__main__":
    # Example usage
    from src.data.generators import IAMDataGenerator
    from src.data.preprocessors import IAMDataPreprocessor

    print("Generating and preprocessing data...")
    generator = IAMDataGenerator()
    df = generator.generate_complete_dataset(
        num_users=50,
        normal_events_per_user=30,
        output_path='data/test_features.csv'
    )

    preprocessor = IAMDataPreprocessor()
    df = preprocessor.preprocess_for_training(df)

    print("\nExtracting features...")
    extractor = IAMFeatureExtractor()
    featured_df = extractor.extract_all_features(df)

    print("\nFeature columns added:")
    new_cols = set(featured_df.columns) - set(df.columns)
    print(new_cols)

    print("\nSample feature values:")
    print(featured_df[list(new_cols)].head())
