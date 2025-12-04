"""
CrowdStrike Falcon Event Parser.

v1.1 Enhancement - December 2025

Parses and normalizes CrowdStrike Falcon ITDR events into a format
compatible with AI Access Sentinel's ML pipeline.

Why This Module?
- Falcon events have different schema than our internal AccessEvent
- Need to extract relevant identity fields for correlation
- Maps Falcon alert types to our risk categories
- Enables unified threat view across ML + Falcon detections
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class FalconAlertType(Enum):
    """CrowdStrike Falcon ITDR alert types."""
    CREDENTIAL_THEFT = "credential_theft"
    CREDENTIAL_COMPROMISE = "credential_compromise"
    LATERAL_MOVEMENT = "lateral_movement"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    IMPOSSIBLE_TRAVEL = "impossible_travel"
    BRUTE_FORCE = "brute_force"
    PASSWORD_SPRAY = "password_spray"
    MFA_BYPASS = "mfa_bypass"
    SESSION_HIJACK = "session_hijack"
    GOLDEN_TICKET = "golden_ticket"
    SILVER_TICKET = "silver_ticket"
    KERBEROASTING = "kerberoasting"
    DCSYNC = "dcsync"
    UNKNOWN = "unknown"


class MitreAttackTactic(Enum):
    """MITRE ATT&CK Tactics relevant to identity threats."""
    INITIAL_ACCESS = "TA0001"
    EXECUTION = "TA0002"
    PERSISTENCE = "TA0003"
    PRIVILEGE_ESCALATION = "TA0004"
    DEFENSE_EVASION = "TA0005"
    CREDENTIAL_ACCESS = "TA0006"
    DISCOVERY = "TA0007"
    LATERAL_MOVEMENT = "TA0008"
    COLLECTION = "TA0009"
    EXFILTRATION = "TA0010"
    IMPACT = "TA0040"


@dataclass
class NormalizedFalconEvent:
    """
    Normalized Falcon event for AI Access Sentinel integration.

    This class maps Falcon ITDR events to our internal format,
    enabling correlation with ML-detected anomalies.
    """
    # Core identifiers
    event_id: str
    event_type: FalconAlertType
    timestamp: datetime

    # User information
    user_id: Optional[str] = None
    username: Optional[str] = None
    user_principal_name: Optional[str] = None
    department: Optional[str] = None

    # Network information
    source_ip: Optional[str] = None
    source_hostname: Optional[str] = None
    destination_ip: Optional[str] = None
    destination_hostname: Optional[str] = None

    # Threat context
    severity: str = "medium"
    confidence: float = 0.5
    risk_score: float = 50.0
    indicators: List[str] = field(default_factory=list)

    # MITRE ATT&CK mapping
    tactics: List[str] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)

    # Action taken
    action_taken: str = "detected"  # detected, blocked, quarantined
    remediation_required: bool = False

    # Original data
    raw_event: Dict[str, Any] = field(default_factory=dict)

    def to_access_event_format(self) -> Dict[str, Any]:
        """
        Convert to AI Access Sentinel AccessEvent format.

        This enables Falcon events to be processed by our ML models
        alongside regular IAM logs.
        """
        return {
            'user_id': self.user_id or self.username or 'unknown',
            'username': self.username or self.user_principal_name,
            'department': self.department or 'Unknown',
            'resource': f"falcon_alert:{self.event_type.value}",
            'action': 'security_event',
            'timestamp': self.timestamp.isoformat(),
            'source_ip': self.source_ip or '0.0.0.0',
            'location': 'CrowdStrike Falcon',
            'success': self.action_taken != 'blocked',
            # Extended fields for Falcon context
            'falcon_event_id': self.event_id,
            'falcon_severity': self.severity,
            'falcon_confidence': self.confidence,
            'falcon_risk_score': self.risk_score,
            'falcon_indicators': self.indicators,
            'falcon_tactics': self.tactics,
            'falcon_techniques': self.techniques
        }


class FalconEventParser:
    """
    Parser for CrowdStrike Falcon ITDR events.

    Transforms raw Falcon API responses into normalized events
    that can be correlated with AI Access Sentinel detections.

    Usage:
        parser = FalconEventParser()
        normalized = parser.parse_identity_alert(raw_falcon_alert)
        access_event = normalized.to_access_event_format()
    """

    # Mapping of Falcon alert types to our internal types
    ALERT_TYPE_MAP = {
        'CredentialTheft': FalconAlertType.CREDENTIAL_THEFT,
        'CredentialCompromise': FalconAlertType.CREDENTIAL_COMPROMISE,
        'LateralMovement': FalconAlertType.LATERAL_MOVEMENT,
        'PrivilegeEscalation': FalconAlertType.PRIVILEGE_ESCALATION,
        'ImpossibleTravel': FalconAlertType.IMPOSSIBLE_TRAVEL,
        'BruteForce': FalconAlertType.BRUTE_FORCE,
        'PasswordSpray': FalconAlertType.PASSWORD_SPRAY,
        'MFABypass': FalconAlertType.MFA_BYPASS,
        'SessionHijack': FalconAlertType.SESSION_HIJACK,
        'GoldenTicket': FalconAlertType.GOLDEN_TICKET,
        'SilverTicket': FalconAlertType.SILVER_TICKET,
        'Kerberoasting': FalconAlertType.KERBEROASTING,
        'DCSync': FalconAlertType.DCSYNC
    }

    # MITRE ATT&CK technique to tactic mapping
    TECHNIQUE_TACTIC_MAP = {
        'T1110': MitreAttackTactic.CREDENTIAL_ACCESS,  # Brute Force
        'T1078': MitreAttackTactic.PRIVILEGE_ESCALATION,  # Valid Accounts
        'T1021': MitreAttackTactic.LATERAL_MOVEMENT,  # Remote Services
        'T1558': MitreAttackTactic.CREDENTIAL_ACCESS,  # Steal or Forge Kerberos Tickets
        'T1003': MitreAttackTactic.CREDENTIAL_ACCESS,  # OS Credential Dumping
        'T1550': MitreAttackTactic.LATERAL_MOVEMENT,  # Use Alternate Auth Material
        'T1098': MitreAttackTactic.PERSISTENCE,  # Account Manipulation
        'T1136': MitreAttackTactic.PERSISTENCE,  # Create Account
        'T1087': MitreAttackTactic.DISCOVERY,  # Account Discovery
        'T1557': MitreAttackTactic.CREDENTIAL_ACCESS,  # LLMNR/NBT-NS Poisoning
    }

    # Severity to risk score base mapping
    SEVERITY_RISK_MAP = {
        'critical': 95,
        'high': 80,
        'medium': 60,
        'low': 40,
        'informational': 20
    }

    def __init__(self):
        """Initialize the parser."""
        self._parsed_count = 0
        self._error_count = 0

    def parse_identity_alert(
        self,
        raw_alert: Dict[str, Any]
    ) -> NormalizedFalconEvent:
        """
        Parse a raw Falcon identity protection alert.

        Args:
            raw_alert: Raw alert from Falcon API

        Returns:
            NormalizedFalconEvent ready for correlation
        """
        try:
            # Extract event type
            raw_type = raw_alert.get('type') or raw_alert.get('alertType') or 'unknown'
            event_type = self.ALERT_TYPE_MAP.get(raw_type, FalconAlertType.UNKNOWN)

            # Extract timestamp
            timestamp_str = raw_alert.get('timestamp') or raw_alert.get('created_timestamp')
            if timestamp_str:
                timestamp = self._parse_timestamp(timestamp_str)
            else:
                timestamp = datetime.utcnow()

            # Extract user information
            user_info = raw_alert.get('user', {}) or {}
            user_id = user_info.get('id') or raw_alert.get('user_id')
            username = user_info.get('name') or raw_alert.get('username')
            upn = user_info.get('userPrincipalName') or raw_alert.get('user_principal_name')

            # Extract network information
            source_ip = raw_alert.get('source_ip') or raw_alert.get('sourceIp')
            dest_ip = raw_alert.get('destination_ip') or raw_alert.get('destinationIp')

            # Extract severity and confidence
            severity = (raw_alert.get('severity') or 'medium').lower()
            confidence = raw_alert.get('confidence') or raw_alert.get('confidenceScore') or 0.5
            if isinstance(confidence, (int, float)) and confidence > 1:
                confidence = confidence / 100  # Normalize to 0-1

            # Calculate risk score
            risk_score = self._calculate_risk_score(severity, confidence, event_type)

            # Extract MITRE ATT&CK data
            tactics = raw_alert.get('tactics', [])
            techniques = raw_alert.get('techniques', [])

            if not tactics and techniques:
                # Derive tactics from techniques
                tactics = self._derive_tactics(techniques)

            # Extract indicators
            indicators = raw_alert.get('indicators', []) or []
            if isinstance(indicators, str):
                indicators = [indicators]

            normalized = NormalizedFalconEvent(
                event_id=raw_alert.get('id') or raw_alert.get('alertId') or f"falcon_{self._parsed_count}",
                event_type=event_type,
                timestamp=timestamp,
                user_id=user_id,
                username=username,
                user_principal_name=upn,
                source_ip=source_ip,
                destination_ip=dest_ip,
                severity=severity,
                confidence=confidence,
                risk_score=risk_score,
                indicators=indicators,
                tactics=tactics,
                techniques=techniques,
                action_taken=raw_alert.get('action', 'detected'),
                remediation_required=severity in ['critical', 'high'],
                raw_event=raw_alert
            )

            self._parsed_count += 1
            return normalized

        except Exception as e:
            self._error_count += 1
            logger.error(f"Error parsing Falcon alert: {e}")

            # Return a minimal event on error
            return NormalizedFalconEvent(
                event_id=f"error_{self._error_count}",
                event_type=FalconAlertType.UNKNOWN,
                timestamp=datetime.utcnow(),
                raw_event=raw_alert
            )

    def parse_incident(
        self,
        raw_incident: Dict[str, Any]
    ) -> List[NormalizedFalconEvent]:
        """
        Parse a Falcon incident which may contain multiple alerts.

        Args:
            raw_incident: Raw incident from Falcon API

        Returns:
            List of normalized events from the incident
        """
        events = []

        try:
            # Incidents may contain multiple alerts
            alerts = raw_incident.get('alerts', []) or [raw_incident]

            for alert in alerts:
                # Inherit incident-level data
                alert['incident_id'] = raw_incident.get('incident_id')
                alert['severity'] = alert.get('severity') or raw_incident.get('fine_score_severity')

                event = self.parse_identity_alert(alert)
                events.append(event)

        except Exception as e:
            logger.error(f"Error parsing Falcon incident: {e}")

        return events

    def parse_webhook_payload(
        self,
        payload: Dict[str, Any]
    ) -> List[NormalizedFalconEvent]:
        """
        Parse a Falcon webhook payload.

        Webhook payloads may contain detection, incident, or identity events.

        Args:
            payload: Raw webhook payload

        Returns:
            List of normalized events
        """
        events = []

        try:
            event_type = payload.get('metadata', {}).get('eventType', '')

            if 'detection' in event_type.lower():
                events.append(self.parse_identity_alert(payload))
            elif 'incident' in event_type.lower():
                events.extend(self.parse_incident(payload))
            elif 'identity' in event_type.lower():
                events.append(self.parse_identity_alert(payload))
            else:
                # Generic parsing
                events.append(self.parse_identity_alert(payload))

        except Exception as e:
            logger.error(f"Error parsing webhook payload: {e}")

        return events

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse various timestamp formats from Falcon."""
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S"
        ]

        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue

        # If all formats fail, return current time
        logger.warning(f"Could not parse timestamp: {timestamp_str}")
        return datetime.utcnow()

    def _calculate_risk_score(
        self,
        severity: str,
        confidence: float,
        event_type: FalconAlertType
    ) -> float:
        """
        Calculate risk score based on severity, confidence, and event type.

        Risk Score Formula:
            base_score = SEVERITY_RISK_MAP[severity]
            type_modifier = 1.0 to 1.2 based on attack type criticality
            final_score = base_score * confidence * type_modifier
        """
        base_score = self.SEVERITY_RISK_MAP.get(severity, 50)

        # Event type modifiers - some attacks are inherently more critical
        type_modifiers = {
            FalconAlertType.GOLDEN_TICKET: 1.2,
            FalconAlertType.DCSYNC: 1.2,
            FalconAlertType.LATERAL_MOVEMENT: 1.15,
            FalconAlertType.PRIVILEGE_ESCALATION: 1.1,
            FalconAlertType.CREDENTIAL_THEFT: 1.1,
            FalconAlertType.SESSION_HIJACK: 1.1,
        }

        modifier = type_modifiers.get(event_type, 1.0)
        final_score = base_score * confidence * modifier

        return min(100, round(final_score, 2))

    def _derive_tactics(self, techniques: List[str]) -> List[str]:
        """Derive MITRE ATT&CK tactics from techniques."""
        tactics = set()

        for technique in techniques:
            # Handle sub-techniques (e.g., T1110.001)
            base_technique = technique.split('.')[0]

            if base_technique in self.TECHNIQUE_TACTIC_MAP:
                tactic = self.TECHNIQUE_TACTIC_MAP[base_technique]
                tactics.add(tactic.value)

        return list(tactics)

    def get_stats(self) -> Dict[str, int]:
        """Get parser statistics."""
        return {
            'parsed_count': self._parsed_count,
            'error_count': self._error_count
        }


def extract_user_identifiers(
    falcon_event: NormalizedFalconEvent
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract all user identifiers from a Falcon event.

    Returns:
        Tuple of (user_id, username, user_principal_name)
    """
    return (
        falcon_event.user_id,
        falcon_event.username,
        falcon_event.user_principal_name
    )


def map_falcon_to_ai_sentinel_risk_level(
    falcon_severity: str
) -> str:
    """
    Map Falcon severity to AI Access Sentinel risk levels.

    Falcon: critical, high, medium, low, informational
    AI Sentinel: CRITICAL, HIGH, MEDIUM, LOW, MINIMAL
    """
    mapping = {
        'critical': 'CRITICAL',
        'high': 'HIGH',
        'medium': 'MEDIUM',
        'low': 'LOW',
        'informational': 'MINIMAL'
    }
    return mapping.get(falcon_severity.lower(), 'MEDIUM')
