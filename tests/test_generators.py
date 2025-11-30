"""
Tests for data generation module.
"""

import pytest
import pandas as pd
from src.data.generators import IAMDataGenerator


def test_iam_generator_initialization():
    """Test IAM data generator initialization."""
    generator = IAMDataGenerator()

    assert len(generator.departments) > 0
    assert len(generator.resources) > 0
    assert 'low_sensitivity' in generator.resources
    assert 'high_sensitivity' in generator.resources


def test_generate_users():
    """Test user generation."""
    generator = IAMDataGenerator()
    users_df = generator.generate_users(num_users=50)

    assert len(users_df) == 50
    assert 'user_id' in users_df.columns
    assert 'department' in users_df.columns
    assert 'job_title' in users_df.columns
    assert users_df['user_id'].is_unique


def test_generate_normal_access_pattern():
    """Test normal access pattern generation."""
    generator = IAMDataGenerator()
    users_df = generator.generate_users(num_users=1)
    user = users_df.iloc[0].to_dict()

    from datetime import datetime
    events = generator.generate_normal_access_pattern(
        user,
        start_date=datetime.now(),
        num_events=20
    )

    assert len(events) == 20
    assert all(not event['is_anomaly'] for event in events)
    assert all(event['user_id'] == user['user_id'] for event in events)


def test_generate_anomalous_access_pattern():
    """Test anomalous pattern generation."""
    generator = IAMDataGenerator()
    users_df = generator.generate_users(num_users=1)
    user = users_df.iloc[0].to_dict()

    from datetime import datetime
    events = generator.generate_anomalous_access_pattern(
        user,
        start_date=datetime.now(),
        num_events=5
    )

    assert len(events) == 5
    assert all(event['is_anomaly'] for event in events)
    assert all(event['anomaly_type'] is not None for event in events)


def test_generate_complete_dataset():
    """Test complete dataset generation."""
    generator = IAMDataGenerator()

    df = generator.generate_complete_dataset(
        num_users=30,
        normal_events_per_user=10,
        anomaly_ratio=0.1,
        output_path='data/test_complete.csv'
    )

    assert len(df) > 0
    assert 'event_id' in df.columns
    assert 'user_id' in df.columns
    assert 'resource' in df.columns
    assert 'is_anomaly' in df.columns

    # Check anomaly ratio is reasonable
    anomaly_ratio = df['is_anomaly'].sum() / len(df)
    assert 0.01 <= anomaly_ratio <= 0.20


def test_data_consistency():
    """Test generated data has consistent fields."""
    generator = IAMDataGenerator()

    df = generator.generate_complete_dataset(
        num_users=10,
        normal_events_per_user=5,
        output_path='data/test_consistency.csv'
    )

    # Check required columns exist
    required_cols = [
        'user_id', 'resource', 'action', 'timestamp',
        'location', 'success', 'is_anomaly'
    ]

    for col in required_cols:
        assert col in df.columns

    # Check no null values in critical fields
    assert not df['user_id'].isna().any()
    assert not df['resource'].isna().any()
    assert not df['action'].isna().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
