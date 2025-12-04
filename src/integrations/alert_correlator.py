"""
Alert Correlation Engine.

v1.1 Enhancement - December 2025

Correlates AI Access Sentinel ML-detected anomalies with CrowdStrike Falcon
ITDR alerts to produce high-confidence, actionable security insights.

Why Correlation Matters?
- ML models detect statistical anomalies but may lack threat context
- Falcon provides threat intelligence but may miss subtle patterns
- Combining both reduces false positives and increases detection confidence
- Provides comprehensive evidence for incident response

Correlation Strategy:
1. Time-based: Events within 5-minute window
2. User-based: Same user_id, username, or UPN
3. IP-based: Same source IP address
4. Pattern-based: Related attack patterns (e.g., brute force + credential theft)
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from .falcon_event_parser import (
    NormalizedFalconEvent,
    FalconAlertType,
    map_falcon_to_ai_sentinel_risk_level
)

logger = logging.getLogger(__name__)


class CorrelationConfidence(Enum):
    """Confidence levels for alert correlation."""
    VERY_HIGH = "very_high"  # 90-100%: Multiple strong correlations
    HIGH = "high"            # 75-89%: Clear correlation with evidence
    MEDIUM = "medium"        # 50-74%: Probable correlation
    LOW = "low"              # 25-49%: Possible correlation
    MINIMAL = "minimal"      # 0-24%: Weak or no correlation


@dataclass
class AIAccessSentinelAlert:
    """
    Internal ML-detected anomaly from AI Access Sentinel.

    This represents an anomaly detected by our ML models
    (Isolation Forest, LSTM, etc.)
    """
    alert_id: str
    user_id: str
    username: Optional[str] = None
    department: Optional[str] = None
    resource: str = ""
    action: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_ip: str = ""
    location: str = ""
    is_anomaly: bool = True
    anomaly_score: float = 0.0
    risk_score: float = 0.0
    risk_level: str = "MEDIUM"
    reasons: List[str] = field(default_factory=list)
    model_name: str = "isolation_forest"


@dataclass
class CorrelatedAlert:
    """
    Result of correlating AI Sentinel and Falcon alerts.

    This is the primary output of the correlation engine,
    combining evidence from both detection sources.
    """
    correlation_id: str
    timestamp: datetime
    confidence: CorrelationConfidence
    confidence_score: float  # 0-100

    # Source alerts
    ai_sentinel_alert: Optional[AIAccessSentinelAlert] = None
    falcon_alert: Optional[NormalizedFalconEvent] = None

    # Combined analysis
    user_id: str = ""
    username: Optional[str] = None
    source_ip: str = ""

    # Risk assessment
    combined_risk_score: float = 0.0
    risk_level: str = "MEDIUM"
    severity: str = "medium"

    # Evidence
    correlation_reasons: List[str] = field(default_factory=list)
    mitre_techniques: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    auto_remediate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            'correlation_id': self.correlation_id,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence.value,
            'confidence_score': self.confidence_score,
            'user_id': self.user_id,
            'username': self.username,
            'source_ip': self.source_ip,
            'combined_risk_score': self.combined_risk_score,
            'risk_level': self.risk_level,
            'severity': self.severity,
            'correlation_reasons': self.correlation_reasons,
            'mitre_techniques': self.mitre_techniques,
            'indicators': self.indicators,
            'recommendations': self.recommendations,
            'auto_remediate': self.auto_remediate,
            'ai_sentinel_alert': {
                'alert_id': self.ai_sentinel_alert.alert_id,
                'risk_score': self.ai_sentinel_alert.risk_score,
                'reasons': self.ai_sentinel_alert.reasons
            } if self.ai_sentinel_alert else None,
            'falcon_alert': self.falcon_alert.to_dict() if self.falcon_alert else None
        }


class AlertCorrelator:
    """
    Correlates AI Access Sentinel ML detections with CrowdStrike Falcon alerts.

    The correlator uses multiple strategies to identify related alerts:
    1. User identity matching (user_id, username, UPN)
    2. Temporal proximity (within configurable time window)
    3. Network indicator matching (source IP)
    4. Attack pattern correlation (related MITRE techniques)

    Usage:
        correlator = AlertCorrelator()

        # Add alerts from both sources
        correlator.add_ai_sentinel_alert(ml_detection)
        correlator.add_falcon_alert(falcon_event)

        # Get correlated alerts
        correlated = correlator.correlate()

        # Or correlate a single new event
        result = correlator.correlate_event(new_event, falcon_alerts)
    """

    def __init__(
        self,
        time_window_minutes: int = 5,
        min_confidence_threshold: float = 0.5
    ):
        """
        Initialize the correlator.

        Args:
            time_window_minutes: Time window for temporal correlation
            min_confidence_threshold: Minimum confidence to report correlation
        """
        self.time_window = timedelta(minutes=time_window_minutes)
        self.min_confidence = min_confidence_threshold

        # Alert buffers
        self._ai_alerts: List[AIAccessSentinelAlert] = []
        self._falcon_alerts: List[NormalizedFalconEvent] = []

        # Correlation cache to avoid duplicates
        self._correlation_cache: Dict[str, CorrelatedAlert] = {}

        # Statistics
        self._stats = {
            'total_correlations': 0,
            'high_confidence_correlations': 0,
            'false_positive_reductions': 0
        }

        logger.info(f"AlertCorrelator initialized (time_window={time_window_minutes}min)")

    def add_ai_sentinel_alert(self, alert: AIAccessSentinelAlert):
        """Add an AI Access Sentinel ML detection to the buffer."""
        self._ai_alerts.append(alert)
        self._cleanup_old_alerts()

    def add_falcon_alert(self, alert: NormalizedFalconEvent):
        """Add a CrowdStrike Falcon alert to the buffer."""
        self._falcon_alerts.append(alert)
        self._cleanup_old_alerts()

    def correlate(self) -> List[CorrelatedAlert]:
        """
        Correlate all buffered alerts.

        Returns:
            List of correlated alerts meeting confidence threshold
        """
        correlated = []

        for ai_alert in self._ai_alerts:
            for falcon_alert in self._falcon_alerts:
                correlation = self._correlate_pair(ai_alert, falcon_alert)

                if correlation and correlation.confidence_score >= self.min_confidence * 100:
                    # Check cache to avoid duplicates
                    if correlation.correlation_id not in self._correlation_cache:
                        self._correlation_cache[correlation.correlation_id] = correlation
                        correlated.append(correlation)
                        self._stats['total_correlations'] += 1

                        if correlation.confidence == CorrelationConfidence.VERY_HIGH:
                            self._stats['high_confidence_correlations'] += 1

        return correlated

    def correlate_event(
        self,
        ai_alert: AIAccessSentinelAlert,
        falcon_alerts: List[NormalizedFalconEvent]
    ) -> Optional[CorrelatedAlert]:
        """
        Correlate a single AI Sentinel alert against Falcon alerts.

        This is useful for real-time correlation as events arrive.

        Args:
            ai_alert: New ML detection to correlate
            falcon_alerts: List of recent Falcon alerts

        Returns:
            Best correlation match, or None if no strong correlation
        """
        best_correlation = None
        best_score = 0

        for falcon_alert in falcon_alerts:
            correlation = self._correlate_pair(ai_alert, falcon_alert)

            if correlation and correlation.confidence_score > best_score:
                best_score = correlation.confidence_score
                best_correlation = correlation

        if best_correlation and best_score >= self.min_confidence * 100:
            self._stats['total_correlations'] += 1
            return best_correlation

        return None

    def enrich_ai_alert_with_falcon(
        self,
        ai_alert: AIAccessSentinelAlert,
        falcon_alerts: List[NormalizedFalconEvent]
    ) -> Dict[str, Any]:
        """
        Enrich an AI Sentinel alert with Falcon threat intelligence.

        Even without strong correlation, Falcon data can provide context.

        Args:
            ai_alert: ML detection to enrich
            falcon_alerts: Available Falcon alerts for context

        Returns:
            Enrichment data including threat context and risk adjustment
        """
        enrichment = {
            'original_risk_score': ai_alert.risk_score,
            'falcon_context_found': False,
            'risk_adjustment': 0,
            'adjusted_risk_score': ai_alert.risk_score,
            'threat_indicators': [],
            'related_techniques': [],
            'enrichment_reasons': []
        }

        # Look for any Falcon alerts for this user
        user_falcon_alerts = [
            fa for fa in falcon_alerts
            if self._match_user(ai_alert, fa)
        ]

        if user_falcon_alerts:
            enrichment['falcon_context_found'] = True
            enrichment['enrichment_reasons'].append(
                f"Found {len(user_falcon_alerts)} Falcon alert(s) for user"
            )

            # Aggregate threat intelligence
            for fa in user_falcon_alerts:
                enrichment['threat_indicators'].extend(fa.indicators)
                enrichment['related_techniques'].extend(fa.techniques)

                # Risk adjustment based on Falcon severity
                severity_boost = {
                    'critical': 20,
                    'high': 15,
                    'medium': 10,
                    'low': 5
                }.get(fa.severity, 0)

                enrichment['risk_adjustment'] = max(
                    enrichment['risk_adjustment'],
                    severity_boost
                )

            # Apply risk adjustment
            enrichment['adjusted_risk_score'] = min(
                100,
                ai_alert.risk_score + enrichment['risk_adjustment']
            )

            # Deduplicate
            enrichment['threat_indicators'] = list(set(enrichment['threat_indicators']))
            enrichment['related_techniques'] = list(set(enrichment['related_techniques']))

        return enrichment

    def _correlate_pair(
        self,
        ai_alert: AIAccessSentinelAlert,
        falcon_alert: NormalizedFalconEvent
    ) -> Optional[CorrelatedAlert]:
        """
        Correlate a pair of alerts and calculate confidence.

        Correlation Scoring:
        - User match: +30 points
        - Time proximity (<5 min): +25 points
        - IP match: +20 points
        - Attack pattern match: +15 points
        - Same resource/target: +10 points
        """
        score = 0
        reasons = []

        # 1. User identity matching (most important)
        if self._match_user(ai_alert, falcon_alert):
            score += 30
            reasons.append("User identity match")

        # 2. Temporal proximity
        time_diff = abs((ai_alert.timestamp - falcon_alert.timestamp).total_seconds())
        if time_diff <= self.time_window.total_seconds():
            # Score higher for closer events
            time_score = 25 * (1 - time_diff / self.time_window.total_seconds())
            score += time_score
            reasons.append(f"Time proximity ({int(time_diff)}s apart)")

        # 3. IP address matching
        if ai_alert.source_ip and falcon_alert.source_ip:
            if ai_alert.source_ip == falcon_alert.source_ip:
                score += 20
                reasons.append("Source IP match")

        # 4. Attack pattern correlation
        pattern_score, pattern_reason = self._match_attack_pattern(ai_alert, falcon_alert)
        if pattern_score > 0:
            score += pattern_score
            reasons.append(pattern_reason)

        # 5. Resource/target matching
        if self._match_resource(ai_alert, falcon_alert):
            score += 10
            reasons.append("Related resource/target")

        # Determine confidence level
        if score >= 90:
            confidence = CorrelationConfidence.VERY_HIGH
        elif score >= 75:
            confidence = CorrelationConfidence.HIGH
        elif score >= 50:
            confidence = CorrelationConfidence.MEDIUM
        elif score >= 25:
            confidence = CorrelationConfidence.LOW
        else:
            confidence = CorrelationConfidence.MINIMAL

        # Only return if we have some correlation
        if score < 25:
            return None

        # Calculate combined risk score
        combined_risk = self._calculate_combined_risk(ai_alert, falcon_alert, score)

        # Generate correlation ID
        correlation_id = self._generate_correlation_id(ai_alert, falcon_alert)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            ai_alert, falcon_alert, confidence, combined_risk
        )

        return CorrelatedAlert(
            correlation_id=correlation_id,
            timestamp=datetime.utcnow(),
            confidence=confidence,
            confidence_score=score,
            ai_sentinel_alert=ai_alert,
            falcon_alert=falcon_alert,
            user_id=ai_alert.user_id or falcon_alert.user_id or "",
            username=ai_alert.username or falcon_alert.username,
            source_ip=ai_alert.source_ip or falcon_alert.source_ip or "",
            combined_risk_score=combined_risk,
            risk_level=self._determine_risk_level(combined_risk),
            severity=falcon_alert.severity,
            correlation_reasons=reasons,
            mitre_techniques=falcon_alert.techniques,
            indicators=falcon_alert.indicators,
            recommendations=recommendations,
            auto_remediate=combined_risk >= 90 and confidence in [
                CorrelationConfidence.VERY_HIGH,
                CorrelationConfidence.HIGH
            ]
        )

    def _match_user(
        self,
        ai_alert: AIAccessSentinelAlert,
        falcon_alert: NormalizedFalconEvent
    ) -> bool:
        """Check if alerts reference the same user."""
        # Direct user_id match
        if ai_alert.user_id and falcon_alert.user_id:
            if ai_alert.user_id == falcon_alert.user_id:
                return True

        # Username match
        if ai_alert.username and falcon_alert.username:
            if ai_alert.username.lower() == falcon_alert.username.lower():
                return True

        # UPN match (email-style identifier)
        if ai_alert.username and falcon_alert.user_principal_name:
            if ai_alert.username.lower() == falcon_alert.user_principal_name.lower():
                return True

        return False

    def _match_attack_pattern(
        self,
        ai_alert: AIAccessSentinelAlert,
        falcon_alert: NormalizedFalconEvent
    ) -> Tuple[int, str]:
        """
        Check if attack patterns are related.

        Returns:
            Tuple of (score, reason)
        """
        # Map AI Sentinel anomaly reasons to potential attack types
        ai_patterns = set()
        for reason in ai_alert.reasons:
            reason_lower = reason.lower()
            if 'unusual time' in reason_lower or 'after hours' in reason_lower:
                ai_patterns.add('temporal_anomaly')
            if 'location' in reason_lower or 'travel' in reason_lower:
                ai_patterns.add('geographic_anomaly')
            if 'sensitive' in reason_lower or 'privilege' in reason_lower:
                ai_patterns.add('privilege_anomaly')
            if 'failed' in reason_lower:
                ai_patterns.add('auth_failure')
            if 'peer' in reason_lower or 'deviation' in reason_lower:
                ai_patterns.add('behavioral_anomaly')

        # Map Falcon alert types to patterns
        falcon_patterns = set()
        if falcon_alert.event_type in [
            FalconAlertType.IMPOSSIBLE_TRAVEL
        ]:
            falcon_patterns.add('geographic_anomaly')

        if falcon_alert.event_type in [
            FalconAlertType.BRUTE_FORCE,
            FalconAlertType.PASSWORD_SPRAY
        ]:
            falcon_patterns.add('auth_failure')

        if falcon_alert.event_type in [
            FalconAlertType.PRIVILEGE_ESCALATION,
            FalconAlertType.GOLDEN_TICKET,
            FalconAlertType.SILVER_TICKET
        ]:
            falcon_patterns.add('privilege_anomaly')

        if falcon_alert.event_type in [
            FalconAlertType.CREDENTIAL_THEFT,
            FalconAlertType.CREDENTIAL_COMPROMISE,
            FalconAlertType.KERBEROASTING
        ]:
            falcon_patterns.add('credential_threat')

        if falcon_alert.event_type in [
            FalconAlertType.LATERAL_MOVEMENT
        ]:
            falcon_patterns.add('behavioral_anomaly')

        # Calculate pattern overlap
        overlap = ai_patterns.intersection(falcon_patterns)

        if overlap:
            score = min(15, len(overlap) * 8)
            reason = f"Attack pattern match: {', '.join(overlap)}"
            return score, reason

        return 0, ""

    def _match_resource(
        self,
        ai_alert: AIAccessSentinelAlert,
        falcon_alert: NormalizedFalconEvent
    ) -> bool:
        """Check if alerts target related resources."""
        # This is a simplified check - in production, you'd have
        # more sophisticated resource matching logic
        if ai_alert.resource and falcon_alert.destination_hostname:
            resource_lower = ai_alert.resource.lower()
            target_lower = falcon_alert.destination_hostname.lower()

            # Check for common keywords
            keywords = ['database', 'admin', 'prod', 'server', 'dc', 'domain']
            for keyword in keywords:
                if keyword in resource_lower and keyword in target_lower:
                    return True

        return False

    def _calculate_combined_risk(
        self,
        ai_alert: AIAccessSentinelAlert,
        falcon_alert: NormalizedFalconEvent,
        correlation_score: float
    ) -> float:
        """
        Calculate combined risk score from both sources.

        Formula:
            base = max(ai_risk, falcon_risk)
            boost = correlation_score * 0.1
            combined = min(100, base + boost)
        """
        ai_risk = ai_alert.risk_score
        falcon_risk = falcon_alert.risk_score

        base_risk = max(ai_risk, falcon_risk)
        correlation_boost = correlation_score * 0.1

        combined = min(100, base_risk + correlation_boost)

        return round(combined, 2)

    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from score."""
        if risk_score >= 90:
            return 'CRITICAL'
        elif risk_score >= 75:
            return 'HIGH'
        elif risk_score >= 50:
            return 'MEDIUM'
        elif risk_score >= 25:
            return 'LOW'
        else:
            return 'MINIMAL'

    def _generate_correlation_id(
        self,
        ai_alert: AIAccessSentinelAlert,
        falcon_alert: NormalizedFalconEvent
    ) -> str:
        """Generate unique correlation ID."""
        components = [
            ai_alert.alert_id,
            falcon_alert.event_id,
            ai_alert.user_id or '',
            ai_alert.timestamp.isoformat()
        ]
        hash_input = ':'.join(components)
        return f"CORR-{hashlib.sha256(hash_input.encode()).hexdigest()[:12]}"

    def _generate_recommendations(
        self,
        ai_alert: AIAccessSentinelAlert,
        falcon_alert: NormalizedFalconEvent,
        confidence: CorrelationConfidence,
        combined_risk: float
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Severity-based recommendations
        if combined_risk >= 90:
            recommendations.append("CRITICAL: Immediately disable user account")
            recommendations.append("Initiate incident response protocol")
            recommendations.append("Preserve all logs for forensic analysis")

        elif combined_risk >= 75:
            recommendations.append("Force password reset immediately")
            recommendations.append("Require MFA re-enrollment")
            recommendations.append("Review all recent access activity")

        elif combined_risk >= 50:
            recommendations.append("Enable enhanced monitoring for user")
            recommendations.append("Verify legitimacy of recent actions")
            recommendations.append("Consider temporary access restrictions")

        # Falcon-specific recommendations
        if falcon_alert.event_type == FalconAlertType.CREDENTIAL_THEFT:
            recommendations.append("Check for credential exposure in breach databases")

        if falcon_alert.event_type == FalconAlertType.LATERAL_MOVEMENT:
            recommendations.append("Audit all systems accessed by this user")
            recommendations.append("Check for persistence mechanisms")

        if falcon_alert.event_type == FalconAlertType.PRIVILEGE_ESCALATION:
            recommendations.append("Review recent privilege changes")
            recommendations.append("Audit administrative group memberships")

        # Correlation-specific
        if confidence == CorrelationConfidence.VERY_HIGH:
            recommendations.append(
                "High-confidence correlation: ML anomaly confirmed by Falcon ITDR"
            )

        return recommendations

    def _cleanup_old_alerts(self):
        """Remove alerts outside the time window."""
        cutoff = datetime.utcnow() - (self.time_window * 12)  # Keep 1 hour

        self._ai_alerts = [
            a for a in self._ai_alerts
            if a.timestamp > cutoff
        ]

        self._falcon_alerts = [
            a for a in self._falcon_alerts
            if a.timestamp > cutoff
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get correlator statistics."""
        return {
            **self._stats,
            'buffered_ai_alerts': len(self._ai_alerts),
            'buffered_falcon_alerts': len(self._falcon_alerts),
            'cached_correlations': len(self._correlation_cache)
        }

    def clear_cache(self):
        """Clear correlation cache."""
        self._correlation_cache.clear()
