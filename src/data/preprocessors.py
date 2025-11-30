"""
Data preprocessing utilities.

Handles data cleaning, transformation, and validation for IAM logs.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict


class IAMDataPreprocessor:
    """Preprocess raw IAM logs for ML model training."""

    def __init__(self):
        self.required_columns = [
            'user_id', 'resource', 'action', 'timestamp',
            'location', 'success'
        ]

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has required columns.

        Args:
            df: Input DataFrame

        Returns:
            True if valid, raises ValueError otherwise
        """
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize IAM log data.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Remove duplicates
        df = df.drop_duplicates()

        # Handle missing values
        df = df.dropna(subset=self.required_columns)

        # Standardize timestamp
        if df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Standardize text fields
        df['action'] = df['action'].str.lower()
        df['resource'] = df['resource'].str.lower()

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.

        Args:
            df: Input DataFrame with timestamp column

        Returns:
            DataFrame with temporal features added
        """
        df = df.copy()

        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = df['hour'].between(9, 17).astype(int)

        # Time categories
        df['time_category'] = pd.cut(
            df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )

        return df

    def add_user_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add user-level aggregate features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with user aggregate features
        """
        df = df.copy()

        # Events per user
        user_counts = df.groupby('user_id').size()
        df['user_total_events'] = df['user_id'].map(user_counts)

        # Unique resources per user
        user_resources = df.groupby('user_id')['resource'].nunique()
        df['user_unique_resources'] = df['user_id'].map(user_resources)

        # Success rate per user
        user_success = df.groupby('user_id')['success'].mean()
        df['user_success_rate'] = df['user_id'].map(user_success)

        return df

    def add_resource_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add resource-level features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with resource features
        """
        df = df.copy()

        # Resource sensitivity mapping
        high_sensitivity = [
            'financial_database', 'payroll_system', 'customer_pii',
            'production_database', 'admin_panel', 'security_logs',
            'aws_console', 'payment_processor'
        ]

        medium_sensitivity = [
            'crm_system', 'project_management', 'analytics_dashboard',
            'source_code_repo', 'marketing_platform', 'sales_tools'
        ]

        df['resource_sensitivity'] = df['resource'].apply(
            lambda x: 'high' if x in high_sensitivity
            else 'medium' if x in medium_sensitivity
            else 'low'
        )

        df['resource_sensitivity_score'] = df['resource_sensitivity'].map({
            'high': 3,
            'medium': 2,
            'low': 1
        })

        # Action risk score
        action_risk = {
            'read': 1,
            'execute': 2,
            'download': 2,
            'write': 3,
            'delete': 4,
            'admin': 5
        }

        df['action_risk_score'] = df['action'].map(action_risk).fillna(1)

        # Combined risk score
        df['combined_risk_score'] = (
            df['resource_sensitivity_score'] * df['action_risk_score']
        )

        return df

    def add_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add location-based features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with location features
        """
        df = df.copy()

        # Suspicious locations
        suspicious_locations = [
            'Moscow, RU', 'Beijing, CN', 'Lagos, NG',
            'Tehran, IR', 'Pyongyang, KP'
        ]

        df['is_suspicious_location'] = (
            df['location'].isin(suspicious_locations).astype(int)
        )

        # Location consistency per user
        user_primary_location = df.groupby('user_id')['location'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        )
        df['user_primary_location'] = df['user_id'].map(user_primary_location)
        df['is_different_location'] = (
            df['location'] != df['user_primary_location']
        ).astype(int)

        return df

    def preprocess_for_training(
        self,
        df: pd.DataFrame,
        include_labels: bool = True
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for model training.

        Args:
            df: Raw DataFrame
            include_labels: Whether to include is_anomaly column

        Returns:
            Fully preprocessed DataFrame
        """
        # Validate
        self.validate_data(df)

        # Clean
        df = self.clean_data(df)

        # Add all features
        df = self.add_temporal_features(df)
        df = self.add_user_aggregates(df)
        df = self.add_resource_features(df)
        df = self.add_location_features(df)

        # Drop original timestamp if present (use temporal features instead)
        if 'timestamp' in df.columns:
            df['timestamp_unix'] = df['timestamp'].astype(np.int64) // 10**9

        return df

    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature columns for ML models.

        Returns:
            List of feature column names
        """
        return [
            'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
            'user_total_events', 'user_unique_resources', 'user_success_rate',
            'resource_sensitivity_score', 'action_risk_score',
            'combined_risk_score', 'is_suspicious_location',
            'is_different_location'
        ]


if __name__ == "__main__":
    # Example usage
    from src.data.generators import IAMDataGenerator

    print("Generating sample data...")
    generator = IAMDataGenerator()
    df = generator.generate_complete_dataset(
        num_users=50,
        normal_events_per_user=20,
        output_path='data/test_sample.csv'
    )

    print("\nPreprocessing data...")
    preprocessor = IAMDataPreprocessor()
    processed_df = preprocessor.preprocess_for_training(df)

    print("\nProcessed columns:")
    print(processed_df.columns.tolist())

    print("\nFeature columns for ML:")
    print(preprocessor.get_feature_columns())

    print("\nSample processed data:")
    print(processed_df[preprocessor.get_feature_columns()].head())
