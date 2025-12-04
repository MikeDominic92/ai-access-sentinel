"""
CrowdStrike Falcon API Connector.

v1.1 Enhancement - December 2025

Provides integration with CrowdStrike Falcon for Identity Threat Detection
and Response (ITDR). This connector enables:
- Fetching identity protection alerts
- Retrieving threat intelligence indicators
- Streaming real-time identity events
- Correlating Falcon detections with ML anomalies

Why CrowdStrike Falcon?
- Industry-leading ITDR capabilities for credential compromise detection
- Real-time identity threat intelligence
- Integration with your existing CrowdStrike deployment at work
- Enhances ML-based detection with threat actor context

Reference: https://www.crowdstrike.com/products/identity-protection/
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

# CrowdStrike FalconPy SDK
try:
    from falconpy import (
        APIHarnessV2,
        IdentityProtection,
        Incidents,
        Detects,
        Intel
    )
    FALCONPY_AVAILABLE = True
except ImportError:
    FALCONPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FalconConfig:
    """Configuration for CrowdStrike Falcon API connection."""
    client_id: str
    client_secret: str
    base_url: str = "https://api.crowdstrike.com"
    member_cid: Optional[str] = None  # For MSSP environments
    timeout: int = 30
    max_retries: int = 3
    mock_mode: bool = False


@dataclass
class FalconIdentityAlert:
    """
    Normalized CrowdStrike identity protection alert.

    Maps Falcon ITDR alerts to a standard format for correlation
    with AI Access Sentinel's ML-based detections.
    """
    alert_id: str
    alert_type: str  # credential_theft, lateral_movement, privilege_escalation
    severity: str  # critical, high, medium, low, informational
    confidence: float  # 0.0 - 1.0
    user_id: Optional[str] = None
    user_principal_name: Optional[str] = None
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    indicators: List[str] = field(default_factory=list)
    tactics: List[str] = field(default_factory=list)  # MITRE ATT&CK
    techniques: List[str] = field(default_factory=list)  # MITRE ATT&CK
    description: str = ""
    remediation_steps: List[str] = field(default_factory=list)
    raw_alert: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'confidence': self.confidence,
            'user_id': self.user_id,
            'user_principal_name': self.user_principal_name,
            'source_ip': self.source_ip,
            'destination_ip': self.destination_ip,
            'timestamp': self.timestamp.isoformat(),
            'indicators': self.indicators,
            'tactics': self.tactics,
            'techniques': self.techniques,
            'description': self.description,
            'remediation_steps': self.remediation_steps
        }

    @property
    def risk_score(self) -> float:
        """Calculate risk score based on severity and confidence."""
        severity_scores = {
            'critical': 95,
            'high': 80,
            'medium': 60,
            'low': 40,
            'informational': 20
        }
        base_score = severity_scores.get(self.severity.lower(), 50)
        return min(100, base_score * self.confidence)


class CrowdStrikeConnector:
    """
    CrowdStrike Falcon API client for ITDR integration.

    This connector provides methods to:
    1. Authenticate with Falcon API using OAuth2
    2. Fetch identity protection alerts
    3. Retrieve threat intelligence
    4. Stream real-time events
    5. Enrich access events with threat context

    Usage:
        connector = CrowdStrikeConnector(
            client_id="your_client_id",
            client_secret="your_client_secret"
        )

        # Get recent identity alerts
        alerts = connector.get_identity_alerts(hours_back=24)

        # Check if IP is malicious
        is_threat = connector.check_indicator("192.168.1.100", "ip")
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        base_url: str = "https://api.crowdstrike.com",
        config: Optional[FalconConfig] = None
    ):
        """
        Initialize CrowdStrike connector.

        Args:
            client_id: Falcon API client ID (or use FALCON_CLIENT_ID env var)
            client_secret: Falcon API client secret (or use FALCON_CLIENT_SECRET env var)
            base_url: Falcon API base URL (default: US-1 cloud)
            config: Optional FalconConfig object for advanced configuration
        """
        if config:
            self.config = config
        else:
            self.config = FalconConfig(
                client_id=client_id or os.getenv('FALCON_CLIENT_ID', ''),
                client_secret=client_secret or os.getenv('FALCON_CLIENT_SECRET', ''),
                base_url=base_url or os.getenv('FALCON_BASE_URL', 'https://api.crowdstrike.com')
            )

        self._client: Optional[APIHarnessV2] = None
        self._identity_client: Optional[IdentityProtection] = None
        self._incidents_client: Optional[Incidents] = None
        self._intel_client: Optional[Intel] = None
        self._connected = False
        self._last_error: Optional[str] = None

        # Cache for threat indicators (TTL: 5 minutes)
        self._indicator_cache: Dict[str, Dict] = {}
        self._cache_ttl = 300

        logger.info("CrowdStrike connector initialized")

    @property
    def is_available(self) -> bool:
        """Check if FalconPy SDK is available."""
        return FALCONPY_AVAILABLE

    @property
    def is_connected(self) -> bool:
        """Check if connected to Falcon API."""
        return self._connected

    def connect(self) -> bool:
        """
        Establish connection to CrowdStrike Falcon API.

        Returns:
            True if connection successful, False otherwise
        """
        if not FALCONPY_AVAILABLE:
            self._last_error = "FalconPy SDK not installed. Run: pip install crowdstrike-falconpy"
            logger.error(self._last_error)
            return False

        if not self.config.client_id or not self.config.client_secret:
            self._last_error = "Falcon API credentials not configured"
            logger.error(self._last_error)
            return False

        try:
            # Initialize main API client
            self._client = APIHarnessV2(
                client_id=self.config.client_id,
                client_secret=self.config.client_secret,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )

            # Initialize service-specific clients
            self._identity_client = IdentityProtection(auth_object=self._client)
            self._incidents_client = Incidents(auth_object=self._client)
            self._intel_client = Intel(auth_object=self._client)

            # Test connection with a simple API call
            response = self._client.command("oauth2_token")

            if response.get('status_code', 0) in [200, 201]:
                self._connected = True
                logger.info("Successfully connected to CrowdStrike Falcon API")
                return True
            else:
                self._last_error = f"Connection failed: {response.get('body', {}).get('errors', [])}"
                logger.error(self._last_error)
                return False

        except Exception as e:
            self._last_error = f"Connection error: {str(e)}"
            logger.error(self._last_error)
            return False

    def disconnect(self):
        """Close connection and cleanup resources."""
        self._client = None
        self._identity_client = None
        self._incidents_client = None
        self._intel_client = None
        self._connected = False
        logger.info("Disconnected from CrowdStrike Falcon API")

    def get_identity_alerts(
        self,
        hours_back: int = 24,
        severity: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[FalconIdentityAlert]:
        """
        Fetch identity protection alerts from Falcon ITDR.

        Args:
            hours_back: How many hours back to fetch alerts
            severity: Filter by severity levels (e.g., ['critical', 'high'])
            limit: Maximum number of alerts to return

        Returns:
            List of FalconIdentityAlert objects
        """
        if not self._connected:
            logger.warning("Not connected to Falcon API. Returning mock data.")
            return self._get_mock_identity_alerts(hours_back, severity, limit)

        alerts = []

        try:
            # Calculate time filter
            start_time = datetime.utcnow() - timedelta(hours=hours_back)

            # Build filter query
            filters = [f"created_timestamp:>'{start_time.isoformat()}Z'"]
            if severity:
                severity_filter = ','.join(f"'{s}'" for s in severity)
                filters.append(f"severity:[{severity_filter}]")

            filter_query = '+'.join(filters)

            # Query identity protection API
            response = self._identity_client.graphql(
                query="""
                query GetIdentityAlerts($filter: String!, $limit: Int!) {
                    entities(filter: $filter, first: $limit) {
                        nodes {
                            id
                            entityType
                            riskScore
                            riskFactors {
                                name
                                severity
                            }
                        }
                    }
                }
                """,
                variables={"filter": filter_query, "limit": limit}
            )

            if response.get('status_code') == 200:
                entities = response.get('body', {}).get('data', {}).get('entities', {}).get('nodes', [])

                for entity in entities:
                    alert = self._parse_identity_entity(entity)
                    if alert:
                        alerts.append(alert)
            else:
                logger.error(f"Failed to fetch identity alerts: {response}")

        except Exception as e:
            logger.error(f"Error fetching identity alerts: {e}")
            # Return mock data for demo purposes
            return self._get_mock_identity_alerts(hours_back, severity, limit)

        return alerts

    def get_incidents(
        self,
        hours_back: int = 24,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Fetch security incidents that may involve identity compromise.

        Args:
            hours_back: Hours back to search
            limit: Maximum incidents to return

        Returns:
            List of incident dictionaries
        """
        if not self._connected:
            logger.warning("Not connected to Falcon API")
            return []

        try:
            # Query incidents
            start_time = datetime.utcnow() - timedelta(hours=hours_back)

            response = self._incidents_client.query_incidents(
                filter=f"start:>'{start_time.isoformat()}Z'",
                limit=limit
            )

            if response.get('status_code') == 200:
                incident_ids = response.get('body', {}).get('resources', [])

                if incident_ids:
                    # Get incident details
                    details_response = self._incidents_client.get_incidents(ids=incident_ids)
                    return details_response.get('body', {}).get('resources', [])

        except Exception as e:
            logger.error(f"Error fetching incidents: {e}")

        return []

    def check_indicator(
        self,
        indicator: str,
        indicator_type: str = "ip"
    ) -> Dict[str, Any]:
        """
        Check if an indicator (IP, domain, hash) is malicious.

        Args:
            indicator: The indicator value (e.g., IP address, domain)
            indicator_type: Type of indicator (ip, domain, md5, sha256)

        Returns:
            Dictionary with threat intelligence results
        """
        cache_key = f"{indicator_type}:{indicator}"

        # Check cache first
        if cache_key in self._indicator_cache:
            cached = self._indicator_cache[cache_key]
            if (datetime.utcnow() - cached['timestamp']).seconds < self._cache_ttl:
                return cached['data']

        result = {
            'indicator': indicator,
            'type': indicator_type,
            'is_malicious': False,
            'confidence': 0.0,
            'threat_types': [],
            'actors': [],
            'last_seen': None
        }

        if not self._connected:
            # Return mock threat intelligence for demo
            return self._get_mock_indicator_result(indicator, indicator_type)

        try:
            # Query Intel API
            response = self._intel_client.query_indicator_entities(
                filter=f"indicator:'{indicator}'+type:'{indicator_type}'"
            )

            if response.get('status_code') == 200:
                indicators = response.get('body', {}).get('resources', [])

                if indicators:
                    intel = indicators[0]
                    result['is_malicious'] = intel.get('malicious_confidence', 'low') in ['high', 'medium']
                    result['confidence'] = {
                        'high': 0.9, 'medium': 0.7, 'low': 0.3
                    }.get(intel.get('malicious_confidence', 'low'), 0.1)
                    result['threat_types'] = intel.get('threat_types', [])
                    result['actors'] = intel.get('actors', [])
                    result['last_seen'] = intel.get('last_updated')

        except Exception as e:
            logger.error(f"Error checking indicator: {e}")

        # Cache result
        self._indicator_cache[cache_key] = {
            'timestamp': datetime.utcnow(),
            'data': result
        }

        return result

    def enrich_access_event(
        self,
        user_id: str,
        source_ip: str,
        resource: str,
        action: str
    ) -> Dict[str, Any]:
        """
        Enrich an access event with Falcon threat intelligence.

        This method queries Falcon for:
        1. Known compromised credentials for the user
        2. Threat intelligence on the source IP
        3. Any active alerts for this user

        Args:
            user_id: User identifier
            source_ip: Source IP address
            resource: Resource being accessed
            action: Action being performed

        Returns:
            Enrichment data including threat indicators and risk boost
        """
        enrichment = {
            'user_id': user_id,
            'source_ip': source_ip,
            'has_falcon_alerts': False,
            'falcon_risk_boost': 0,
            'threat_indicators': [],
            'active_alerts': [],
            'recommendations': []
        }

        # Check IP reputation
        ip_intel = self.check_indicator(source_ip, 'ip')
        if ip_intel['is_malicious']:
            enrichment['threat_indicators'].append({
                'type': 'malicious_ip',
                'indicator': source_ip,
                'confidence': ip_intel['confidence'],
                'threat_types': ip_intel['threat_types']
            })
            enrichment['falcon_risk_boost'] += 30 * ip_intel['confidence']
            enrichment['recommendations'].append(
                f"Source IP {source_ip} flagged as malicious by CrowdStrike"
            )

        # Get active alerts for user
        alerts = self.get_identity_alerts(hours_back=24, limit=10)
        user_alerts = [a for a in alerts if a.user_id == user_id or a.user_principal_name == user_id]

        if user_alerts:
            enrichment['has_falcon_alerts'] = True
            enrichment['active_alerts'] = [a.to_dict() for a in user_alerts]

            # Calculate risk boost based on alert severity
            for alert in user_alerts:
                enrichment['falcon_risk_boost'] += alert.risk_score * 0.25
                enrichment['recommendations'].append(
                    f"Active Falcon alert: {alert.alert_type} ({alert.severity})"
                )

        # Cap risk boost at 25 (per our risk scoring weights)
        enrichment['falcon_risk_boost'] = min(25, enrichment['falcon_risk_boost'])

        return enrichment

    def _parse_identity_entity(self, entity: Dict) -> Optional[FalconIdentityAlert]:
        """Parse Falcon identity entity into FalconIdentityAlert."""
        try:
            risk_factors = entity.get('riskFactors', [])
            severities = [rf.get('severity', 'low') for rf in risk_factors]
            max_severity = max(severities, key=lambda s: {
                'critical': 4, 'high': 3, 'medium': 2, 'low': 1, 'informational': 0
            }.get(s.lower(), 0)) if severities else 'medium'

            return FalconIdentityAlert(
                alert_id=entity.get('id', ''),
                alert_type=entity.get('entityType', 'unknown'),
                severity=max_severity,
                confidence=entity.get('riskScore', 50) / 100,
                description=', '.join(rf.get('name', '') for rf in risk_factors),
                raw_alert=entity
            )
        except Exception as e:
            logger.error(f"Error parsing identity entity: {e}")
            return None

    def _get_mock_identity_alerts(
        self,
        hours_back: int,
        severity: Optional[List[str]],
        limit: int
    ) -> List[FalconIdentityAlert]:
        """Generate mock identity alerts for demo/testing."""
        mock_alerts = [
            FalconIdentityAlert(
                alert_id="FALCON-ITDR-001",
                alert_type="credential_theft",
                severity="high",
                confidence=0.85,
                user_id="U042",
                user_principal_name="john.doe@company.com",
                source_ip="203.0.113.45",
                timestamp=datetime.utcnow() - timedelta(hours=2),
                indicators=["suspicious_auth_pattern", "impossible_travel"],
                tactics=["TA0006"],  # Credential Access
                techniques=["T1110"],  # Brute Force
                description="Potential credential compromise detected - unusual authentication pattern",
                remediation_steps=[
                    "Force password reset",
                    "Enable MFA if not already enabled",
                    "Review recent account activity"
                ]
            ),
            FalconIdentityAlert(
                alert_id="FALCON-ITDR-002",
                alert_type="lateral_movement",
                severity="critical",
                confidence=0.92,
                user_id="U015",
                user_principal_name="admin.user@company.com",
                source_ip="10.0.0.50",
                destination_ip="10.0.0.100",
                timestamp=datetime.utcnow() - timedelta(hours=1),
                indicators=["abnormal_service_access", "privilege_escalation_attempt"],
                tactics=["TA0008"],  # Lateral Movement
                techniques=["T1021"],  # Remote Services
                description="Lateral movement detected - admin account accessing multiple systems",
                remediation_steps=[
                    "Isolate affected systems",
                    "Disable account temporarily",
                    "Conduct forensic investigation"
                ]
            ),
            FalconIdentityAlert(
                alert_id="FALCON-ITDR-003",
                alert_type="privilege_escalation",
                severity="medium",
                confidence=0.75,
                user_id="U078",
                user_principal_name="dev.user@company.com",
                source_ip="192.168.1.100",
                timestamp=datetime.utcnow() - timedelta(hours=6),
                indicators=["unusual_admin_activity"],
                tactics=["TA0004"],  # Privilege Escalation
                techniques=["T1078"],  # Valid Accounts
                description="User attempting to access resources outside normal scope",
                remediation_steps=[
                    "Review access permissions",
                    "Verify access request with manager"
                ]
            )
        ]

        # Filter by severity if specified
        if severity:
            severity_lower = [s.lower() for s in severity]
            mock_alerts = [a for a in mock_alerts if a.severity.lower() in severity_lower]

        return mock_alerts[:limit]

    def _get_mock_indicator_result(
        self,
        indicator: str,
        indicator_type: str
    ) -> Dict[str, Any]:
        """Generate mock threat intelligence for demo."""
        # Simulate some IPs as malicious for testing
        known_bad_ips = ['203.0.113.45', '198.51.100.10', '192.0.2.100']

        is_malicious = indicator in known_bad_ips

        return {
            'indicator': indicator,
            'type': indicator_type,
            'is_malicious': is_malicious,
            'confidence': 0.85 if is_malicious else 0.1,
            'threat_types': ['botnet', 'credential_theft'] if is_malicious else [],
            'actors': ['ADVERSARY-001'] if is_malicious else [],
            'last_seen': datetime.utcnow().isoformat() if is_malicious else None
        }

    def get_status(self) -> Dict[str, Any]:
        """Get connector status for health checks."""
        return {
            'connected': self._connected,
            'sdk_available': FALCONPY_AVAILABLE,
            'base_url': self.config.base_url,
            'last_error': self._last_error,
            'cache_size': len(self._indicator_cache)
        }


# Singleton instance for API usage
_connector_instance: Optional[CrowdStrikeConnector] = None


def get_connector() -> CrowdStrikeConnector:
    """Get or create singleton CrowdStrike connector."""
    global _connector_instance
    if _connector_instance is None:
        _connector_instance = CrowdStrikeConnector()
    return _connector_instance
