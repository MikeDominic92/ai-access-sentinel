"""
Risk scoring engine for users and access events.

Combines multiple factors to calculate comprehensive risk scores.

v1.1 Enhancement - December 2025:
- Added CrowdStrike Falcon threat intelligence integration
- New factor: falcon_threat_score (25% weight when enabled)
- Risk weights automatically adjust when Falcon data is available
- Correlation with ITDR alerts for enhanced detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RiskScorer:
    """
    Calculate risk scores for users based on multiple factors.

    Base Factors (v1.0):
    - Anomaly count (30%)
    - Peer deviation (20%)
    - Sensitive resource access (20%)
    - Failed attempts (15%)
    - Policy violations (15%)

    Enhanced Factors (v1.1 with Falcon):
    - Anomaly count (22.5%)
    - Peer deviation (15%)
    - Sensitive resource access (15%)
    - Failed attempts (11.25%)
    - Policy violations (11.25%)
    - Falcon threat intelligence (25%)
    """

    def __init__(self, enable_falcon: bool = True):
        """
        Initialize RiskScorer.

        Args:
            enable_falcon: Enable CrowdStrike Falcon integration (default: True)
        """
        self.enable_falcon = enable_falcon

        # Base weights (without Falcon)
        self._base_weights = {
            'anomaly_score': 0.30,
            'peer_deviation': 0.20,
            'sensitive_access': 0.20,
            'failed_attempts': 0.15,
            'policy_violations': 0.15
        }

        # Enhanced weights (with Falcon - 25% allocated to Falcon)
        self._falcon_weights = {
            'anomaly_score': 0.225,      # 30% * 0.75
            'peer_deviation': 0.15,       # 20% * 0.75
            'sensitive_access': 0.15,     # 20% * 0.75
            'failed_attempts': 0.1125,    # 15% * 0.75
            'policy_violations': 0.1125,  # 15% * 0.75
            'falcon_threat': 0.25         # New: 25%
        }

        # Use appropriate weights
        self.risk_weights = self._falcon_weights if enable_falcon else self._base_weights

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

    def calculate_falcon_threat_factor(
        self,
        user_id: str,
        falcon_alerts: Optional[List[Dict[str, Any]]] = None,
        falcon_enrichment: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate Falcon threat intelligence factor (0-100).

        v1.1 Enhancement - December 2025

        This factor incorporates CrowdStrike Falcon ITDR data:
        - Active identity protection alerts for the user
        - Threat intelligence on source IPs
        - Credential compromise indicators
        - Lateral movement detections

        Args:
            user_id: User to score
            falcon_alerts: List of Falcon alerts for this user
            falcon_enrichment: Pre-computed enrichment data

        Returns:
            Falcon threat factor score (0-100)
        """
        if not self.enable_falcon:
            return 0

        score = 0

        # If we have pre-computed enrichment, use it
        if falcon_enrichment:
            # Direct risk boost from enrichment
            score = falcon_enrichment.get('falcon_risk_boost', 0) * 4  # Scale to 0-100

            # Add for active alerts
            if falcon_enrichment.get('has_falcon_alerts'):
                active_alerts = falcon_enrichment.get('active_alerts', [])
                for alert in active_alerts:
                    severity = alert.get('severity', 'low')
                    severity_score = {
                        'critical': 40,
                        'high': 30,
                        'medium': 20,
                        'low': 10
                    }.get(severity, 5)
                    score += severity_score

            # Add for malicious indicators
            for indicator in falcon_enrichment.get('threat_indicators', []):
                confidence = indicator.get('confidence', 0.5)
                score += 20 * confidence

            return min(100, score)

        # If we have raw Falcon alerts, process them
        if falcon_alerts:
            user_alerts = [
                a for a in falcon_alerts
                if a.get('user_id') == user_id or
                   a.get('user_principal_name', '').lower() == user_id.lower()
            ]

            if not user_alerts:
                return 0

            # Score based on alert severity and confidence
            for alert in user_alerts:
                severity = alert.get('severity', 'medium').lower()
                confidence = alert.get('confidence', 0.5)

                severity_base = {
                    'critical': 50,
                    'high': 35,
                    'medium': 20,
                    'low': 10,
                    'informational': 5
                }.get(severity, 15)

                # Weight by confidence
                alert_score = severity_base * confidence

                # Boost for specific high-risk alert types
                alert_type = alert.get('alert_type', '').lower()
                if alert_type in ['credential_theft', 'golden_ticket', 'dcsync']:
                    alert_score *= 1.3
                elif alert_type in ['lateral_movement', 'privilege_escalation']:
                    alert_score *= 1.2

                score += alert_score

            return min(100, score)

        return 0

    def calculate_user_risk_score(
        self,
        df: pd.DataFrame,
        user_id: str,
        department: str = None,
        falcon_alerts: Optional[List[Dict[str, Any]]] = None,
        falcon_enrichment: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive risk score for a user.

        v1.1 Enhancement: Now integrates CrowdStrike Falcon ITDR data
        for enhanced threat detection and correlation.

        Args:
            df: IAM logs DataFrame
            user_id: User to score
            department: User's department (if known)
            falcon_alerts: CrowdStrike Falcon alerts for this user (v1.1)
            falcon_enrichment: Pre-computed Falcon enrichment data (v1.1)

        Returns:
            Dictionary with risk score, breakdown, and Falcon correlation
        """
        # Get department if not provided
        if department is None and 'department' in df.columns:
            user_rows = df[df['user_id'] == user_id]
            if len(user_rows) > 0:
                department = user_rows['department'].iloc[0]
            else:
                department = 'Unknown'

        # Calculate base factors
        factors = {
            'anomaly_score': self.calculate_anomaly_factor(df, user_id),
            'peer_deviation': self.calculate_peer_deviation_factor(df, user_id, department) if department != 'Unknown' else 0,
            'sensitive_access': self.calculate_sensitive_access_factor(df, user_id),
            'failed_attempts': self.calculate_failed_attempts_factor(df, user_id),
            'policy_violations': self.calculate_policy_violations_factor(df, user_id)
        }

        # v1.1: Add Falcon threat factor if enabled
        falcon_score = 0
        falcon_context = {}
        if self.enable_falcon:
            falcon_score = self.calculate_falcon_threat_factor(
                user_id, falcon_alerts, falcon_enrichment
            )
            factors['falcon_threat'] = falcon_score

            # Capture Falcon context for response
            if falcon_alerts:
                user_alerts = [
                    a for a in falcon_alerts
                    if a.get('user_id') == user_id or
                       a.get('user_principal_name', '').lower() == user_id.lower()
                ]
                falcon_context = {
                    'active_alerts': len(user_alerts),
                    'alert_types': list(set(a.get('alert_type', 'unknown') for a in user_alerts)),
                    'max_severity': max(
                        (a.get('severity', 'low') for a in user_alerts),
                        key=lambda s: {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}.get(s, 0),
                        default='none'
                    ) if user_alerts else 'none'
                }

            logger.debug(f"Falcon threat factor for {user_id}: {falcon_score}")

        # Calculate weighted score using appropriate weights
        total_score = sum(
            factors[key] * self.risk_weights.get(key, 0)
            for key in factors.keys()
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

        # Generate recommendations (includes Falcon-aware recommendations)
        recommendations = self._generate_recommendations(factors, risk_level)

        result = {
            'user_id': user_id,
            'risk_score': round(total_score, 2),
            'risk_level': risk_level,
            'factor_scores': {k: round(v, 2) for k, v in factors.items()},
            'recommendations': recommendations,
            'falcon_enabled': self.enable_falcon
        }

        # Add Falcon context if available
        if falcon_context:
            result['falcon_context'] = falcon_context

        return result

    def _generate_recommendations(
        self,
        factors: Dict[str, float],
        risk_level: str
    ) -> List[str]:
        """
        Generate recommendations based on risk factors.

        v1.1: Now includes Falcon-aware recommendations for
        identity threat response.
        """
        recommendations = []

        if factors.get('anomaly_score', 0) > 50:
            recommendations.append("High anomaly count - increase monitoring")

        if factors.get('peer_deviation', 0) > 60:
            recommendations.append("Unusual access pattern vs peers - review permissions")

        if factors.get('sensitive_access', 0) > 70:
            recommendations.append("Frequent sensitive resource access - enforce MFA")

        if factors.get('failed_attempts', 0) > 40:
            recommendations.append("Multiple failed attempts - investigate credentials")

        if factors.get('policy_violations', 0) > 50:
            recommendations.append("Policy violations detected - conduct security training")

        # v1.1: Falcon-aware recommendations
        falcon_score = factors.get('falcon_threat', 0)
        if falcon_score > 80:
            recommendations.append("CRITICAL: Active Falcon ITDR alert - initiate incident response")
            recommendations.append("Recommend temporary access suspension pending investigation")
        elif falcon_score > 60:
            recommendations.append("HIGH: Falcon threat intelligence indicates compromise risk")
            recommendations.append("Review CrowdStrike Falcon console for detailed indicators")
        elif falcon_score > 40:
            recommendations.append("MEDIUM: Falcon detected suspicious identity activity")
            recommendations.append("Correlate with IAM logs for full context")
        elif falcon_score > 20:
            recommendations.append("LOW: Minor Falcon indicators - continue enhanced monitoring")

        if risk_level in ['CRITICAL', 'HIGH']:
            recommendations.append("Require immediate security review")

        if not recommendations:
            recommendations.append("Continue normal monitoring")

        return recommendations

    def calculate_batch_risk_scores(
        self,
        df: pd.DataFrame,
        falcon_alerts: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        Calculate risk scores for all users.

        v1.1: Now supports Falcon alert correlation across all users.

        Args:
            df: IAM logs DataFrame
            falcon_alerts: CrowdStrike Falcon alerts for correlation (v1.1)

        Returns:
            DataFrame with user risk scores
        """
        print("Calculating risk scores for all users...")
        if self.enable_falcon and falcon_alerts:
            print(f"  - Correlating with {len(falcon_alerts)} Falcon alerts")

        results = []
        users = df['user_id'].unique()

        for user_id in users:
            score_result = self.calculate_user_risk_score(
                df, user_id,
                falcon_alerts=falcon_alerts
            )
            results.append(score_result)

        scores_df = pd.DataFrame(results)
        scores_df = scores_df.sort_values('risk_score', ascending=False)

        print(f"Calculated scores for {len(users)} users")

        # Log high-risk users with Falcon correlation
        if self.enable_falcon:
            falcon_elevated = scores_df[
                scores_df['factor_scores'].apply(
                    lambda x: x.get('falcon_threat', 0) > 50
                )
            ] if 'factor_scores' in scores_df.columns else pd.DataFrame()

            if len(falcon_elevated) > 0:
                logger.warning(
                    f"Found {len(falcon_elevated)} users with elevated Falcon threat scores"
                )

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
