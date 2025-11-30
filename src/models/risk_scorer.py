"""
Risk scoring engine for users and access events.

Combines multiple factors to calculate comprehensive risk scores.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class RiskScorer:
    """
    Calculate risk scores for users based on multiple factors.

    Factors:
    - Anomaly count
    - Peer deviation
    - Sensitive resource access
    - Failed attempts
    - Policy violations
    """

    def __init__(self):
        self.risk_weights = {
            'anomaly_score': 0.30,
            'peer_deviation': 0.20,
            'sensitive_access': 0.20,
            'failed_attempts': 0.15,
            'policy_violations': 0.15
        }

        self.risk_thresholds = {
            'critical': 90,
            'high': 75,
            'medium': 50,
            'low': 25
        }

    def calculate_anomaly_factor(
        self,
        df: pd.DataFrame,
        user_id: str,
        window_days: int = 30
    ) -> float:
        """
        Calculate anomaly factor (0-100).

        Args:
            df: IAM logs with anomaly flags
            user_id: User to score
            window_days: Time window to consider

        Returns:
            Anomaly factor score
        """
        # Filter to recent user events
        cutoff = datetime.now() - timedelta(days=window_days)

        user_events = df[df['user_id'] == user_id]

        if 'timestamp' in df.columns and df['timestamp'].dtype != 'object':
            user_events = user_events[user_events['timestamp'] >= cutoff]

        if len(user_events) == 0:
            return 0

        # Count anomalies
        if 'is_anomaly' in user_events.columns:
            anomaly_count = user_events['is_anomaly'].sum()
            anomaly_ratio = anomaly_count / len(user_events)
        else:
            anomaly_ratio = 0

        # Score: 0-100 based on ratio
        # 0% anomalies = 0, 10% = 50, 20%+ = 100
        score = min(100, anomaly_ratio * 500)

        return score

    def calculate_peer_deviation_factor(
        self,
        df: pd.DataFrame,
        user_id: str,
        department: str
    ) -> float:
        """
        Calculate how much user deviates from peers (0-100).

        Args:
            df: IAM logs
            user_id: User to score
            department: User's department

        Returns:
            Peer deviation score
        """
        # User metrics
        user_data = df[df['user_id'] == user_id]
        user_resource_count = user_data['resource'].nunique()
        user_event_count = len(user_data)

        # Peer metrics
        peer_data = df[
            (df['department'] == department) &
            (df['user_id'] != user_id)
        ]

        if len(peer_data) == 0:
            return 0

        peer_resource_counts = peer_data.groupby('user_id')['resource'].nunique()
        peer_event_counts = peer_data.groupby('user_id').size()

        peer_avg_resources = peer_resource_counts.mean()
        peer_std_resources = peer_resource_counts.std()

        # Calculate z-score
        if peer_std_resources > 0:
            z_score = abs((user_resource_count - peer_avg_resources) / peer_std_resources)
        else:
            z_score = 0

        # Convert to 0-100 score
        # z > 3 = 100, z > 2 = 75, z > 1 = 50
        score = min(100, z_score * 33)

        return score

    def calculate_sensitive_access_factor(
        self,
        df: pd.DataFrame,
        user_id: str
    ) -> float:
        """
        Calculate sensitive resource access score (0-100).

        Args:
            df: IAM logs
            user_id: User to score

        Returns:
            Sensitive access score
        """
        user_data = df[df['user_id'] == user_id]

        if len(user_data) == 0:
            return 0

        # High sensitivity resources
        high_sensitivity = [
            'financial_database', 'payroll_system', 'customer_pii',
            'production_database', 'admin_panel', 'security_logs',
            'aws_console', 'payment_processor'
        ]

        # Count sensitive accesses
        sensitive_count = user_data['resource'].isin(high_sensitivity).sum()
        sensitive_ratio = sensitive_count / len(user_data)

        # Also check for high-risk actions
        high_risk_actions = ['delete', 'admin', 'write']
        risky_action_count = user_data['action'].isin(high_risk_actions).sum()
        risky_action_ratio = risky_action_count / len(user_data)

        # Combined score
        score = (sensitive_ratio * 60) + (risky_action_ratio * 40)
        score = min(100, score * 100)

        return score

    def calculate_failed_attempts_factor(
        self,
        df: pd.DataFrame,
        user_id: str,
        window_days: int = 30
    ) -> float:
        """
        Calculate failed attempt score (0-100).

        Args:
            df: IAM logs
            user_id: User to score
            window_days: Time window

        Returns:
            Failed attempts score
        """
        # Filter to recent user events
        user_events = df[df['user_id'] == user_id]

        if len(user_events) == 0:
            return 0

        # Count failures
        if 'success' in user_events.columns:
            failed_count = (~user_events['success']).sum()
            failure_ratio = failed_count / len(user_events)
        else:
            failure_ratio = 0

        # Score: 0-100
        # 0% failures = 0, 5% = 50, 10%+ = 100
        score = min(100, failure_ratio * 1000)

        return score

    def calculate_policy_violations_factor(
        self,
        df: pd.DataFrame,
        user_id: str
    ) -> float:
        """
        Calculate policy violation score (0-100).

        Args:
            df: IAM logs
            user_id: User to score

        Returns:
            Policy violation score
        """
        user_data = df[df['user_id'] == user_id]

        if len(user_data) == 0:
            return 0

        violation_count = 0

        # Check for violations
        # 1. Access outside business hours
        if 'is_business_hours' in user_data.columns:
            violation_count += (~user_data['is_business_hours'].astype(bool)).sum()

        # 2. Access from suspicious locations
        if 'is_suspicious_location' in user_data.columns:
            violation_count += user_data['is_suspicious_location'].sum()

        # 3. Weekend access to sensitive resources
        if 'is_weekend' in user_data.columns and 'resource_sensitivity_score' in user_data.columns:
            weekend_sensitive = (
                user_data['is_weekend'].astype(bool) &
                (user_data['resource_sensitivity_score'] >= 3)
            ).sum()
            violation_count += weekend_sensitive

        # Calculate ratio
        violation_ratio = violation_count / len(user_data) if len(user_data) > 0 else 0

        # Score: 0-100
        score = min(100, violation_ratio * 500)

        return score

    def calculate_user_risk_score(
        self,
        df: pd.DataFrame,
        user_id: str,
        department: str = None
    ) -> Dict[str, any]:
        """
        Calculate comprehensive risk score for a user.

        Args:
            df: IAM logs DataFrame
            user_id: User to score
            department: User's department (if known)

        Returns:
            Dictionary with risk score and breakdown
        """
        # Get department if not provided
        if department is None and 'department' in df.columns:
            user_rows = df[df['user_id'] == user_id]
            if len(user_rows) > 0:
                department = user_rows['department'].iloc[0]
            else:
                department = 'Unknown'

        # Calculate individual factors
        factors = {
            'anomaly_score': self.calculate_anomaly_factor(df, user_id),
            'peer_deviation': self.calculate_peer_deviation_factor(df, user_id, department) if department != 'Unknown' else 0,
            'sensitive_access': self.calculate_sensitive_access_factor(df, user_id),
            'failed_attempts': self.calculate_failed_attempts_factor(df, user_id),
            'policy_violations': self.calculate_policy_violations_factor(df, user_id)
        }

        # Calculate weighted score
        total_score = sum(
            factors[key] * self.risk_weights[key]
            for key in self.risk_weights.keys()
        )

        # Determine risk level
        if total_score >= self.risk_thresholds['critical']:
            risk_level = 'CRITICAL'
        elif total_score >= self.risk_thresholds['high']:
            risk_level = 'HIGH'
        elif total_score >= self.risk_thresholds['medium']:
            risk_level = 'MEDIUM'
        elif total_score >= self.risk_thresholds['low']:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'

        # Generate recommendations
        recommendations = self._generate_recommendations(factors, risk_level)

        return {
            'user_id': user_id,
            'risk_score': round(total_score, 2),
            'risk_level': risk_level,
            'factor_scores': {k: round(v, 2) for k, v in factors.items()},
            'recommendations': recommendations
        }

    def _generate_recommendations(
        self,
        factors: Dict[str, float],
        risk_level: str
    ) -> List[str]:
        """Generate recommendations based on risk factors."""
        recommendations = []

        if factors['anomaly_score'] > 50:
            recommendations.append("High anomaly count - increase monitoring")

        if factors['peer_deviation'] > 60:
            recommendations.append("Unusual access pattern vs peers - review permissions")

        if factors['sensitive_access'] > 70:
            recommendations.append("Frequent sensitive resource access - enforce MFA")

        if factors['failed_attempts'] > 40:
            recommendations.append("Multiple failed attempts - investigate credentials")

        if factors['policy_violations'] > 50:
            recommendations.append("Policy violations detected - conduct security training")

        if risk_level in ['CRITICAL', 'HIGH']:
            recommendations.append("Require immediate security review")

        if not recommendations:
            recommendations.append("Continue normal monitoring")

        return recommendations

    def calculate_batch_risk_scores(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate risk scores for all users.

        Args:
            df: IAM logs DataFrame

        Returns:
            DataFrame with user risk scores
        """
        print("Calculating risk scores for all users...")

        results = []
        users = df['user_id'].unique()

        for user_id in users:
            score_result = self.calculate_user_risk_score(df, user_id)
            results.append(score_result)

        scores_df = pd.DataFrame(results)
        scores_df = scores_df.sort_values('risk_score', ascending=False)

        print(f"Calculated scores for {len(users)} users")

        return scores_df


if __name__ == "__main__":
    # Example usage
    from src.data.generators import IAMDataGenerator
    from src.data.preprocessors import IAMDataPreprocessor

    # Generate data
    print("Generating data for risk scoring...")
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

    # Calculate risk scores
    print("\n" + "="*50)
    print("Calculating Risk Scores")
    print("="*50)

    scorer = RiskScorer()

    # Single user example
    test_user = df['user_id'].iloc[0]
    result = scorer.calculate_user_risk_score(df, test_user)

    print(f"\nUser: {result['user_id']}")
    print(f"Risk Score: {result['risk_score']}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"\nFactor Breakdown:")
    for factor, score in result['factor_scores'].items():
        print(f"  {factor}: {score:.2f}")
    print(f"\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  - {rec}")

    # Batch scoring
    print("\n" + "="*50)
    print("Top 10 Highest Risk Users")
    print("="*50)

    all_scores = scorer.calculate_batch_risk_scores(df)
    print(all_scores[['user_id', 'risk_score', 'risk_level']].head(10))
