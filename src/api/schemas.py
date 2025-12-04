"""
Pydantic schemas for API request/response validation.

v1.1 Enhancement - December 2025:
- Added CrowdStrike Falcon webhook schemas
- Added Falcon alert correlation response schemas
- Added enhanced risk scoring with Falcon context
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class AccessEventRequest(BaseModel):
    """Request schema for analyzing an access event."""
    user_id: str = Field(..., description="User identifier")
    username: Optional[str] = Field(None, description="Username")
    department: Optional[str] = Field(None, description="User department")
    job_title: Optional[str] = Field(None, description="Job title")
    resource: str = Field(..., description="Resource being accessed")
    action: str = Field(..., description="Action being performed")
    timestamp: datetime = Field(..., description="Event timestamp")
    source_ip: str = Field(..., description="Source IP address")
    location: str = Field(..., description="Access location")
    success: bool = Field(True, description="Whether access was successful")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "U001",
                "username": "john.doe",
                "department": "Engineering",
                "job_title": "Software Engineer",
                "resource": "source_code_repo",
                "action": "read",
                "timestamp": "2024-01-15T14:30:00Z",
                "source_ip": "192.168.1.100",
                "location": "New York, US",
                "success": True
            }
        }


class AnomalyAnalysisResponse(BaseModel):
    """Response schema for anomaly analysis."""
    is_anomaly: bool
    risk_score: float
    anomaly_score: float
    risk_level: str
    reasons: List[str]
    recommendation: str


class BatchAnalysisRequest(BaseModel):
    """Request schema for batch analysis."""
    events: List[AccessEventRequest]


class BatchAnalysisResponse(BaseModel):
    """Response schema for batch analysis."""
    total_events: int
    anomalies_detected: int
    anomaly_ratio: float
    results: List[AnomalyAnalysisResponse]


class UserRiskScoreResponse(BaseModel):
    """Response schema for user risk score."""
    user_id: str
    risk_score: float
    risk_level: str
    factor_scores: Dict[str, float]
    recommendations: List[str]


class RoleDiscoveryResponse(BaseModel):
    """Response schema for role discovery."""
    total_roles: int
    total_users: int
    roles: List[Dict[str, Any]]
    role_health: Dict[str, Any]


class ModelMetricsResponse(BaseModel):
    """Response schema for model metrics."""
    anomaly_detector: Dict[str, float]
    access_predictor: Optional[Dict[str, float]] = None
    role_miner: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    version: str
    models_loaded: Dict[str, bool]
    timestamp: datetime


class AccessPredictionRequest(BaseModel):
    """Request schema for access prediction."""
    user_id: str
    department: str
    job_title: str
    resource: str
    action: str

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "U001",
                "department": "Engineering",
                "job_title": "Software Engineer",
                "resource": "production_database",
                "action": "write"
            }
        }


class AccessPredictionResponse(BaseModel):
    """Response schema for access prediction."""
    should_approve: bool
    confidence: float
    probability_approve: float
    probability_deny: float
    recommendation: str
    peer_analysis: Optional[Dict[str, Any]] = None


# ============================================================================
# v1.1: CrowdStrike Falcon ITDR Integration Schemas
# ============================================================================

class FalconAlertSeverity(str, Enum):
    """CrowdStrike Falcon alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class FalconWebhookEvent(BaseModel):
    """
    Schema for incoming CrowdStrike Falcon webhook events.

    v1.1 Enhancement - December 2025
    Supports identity protection alerts from Falcon ITDR.
    """
    id: Optional[str] = Field(None, description="Falcon event ID")
    alertId: Optional[str] = Field(None, description="Falcon alert ID")
    type: Optional[str] = Field(None, description="Alert type (e.g., CredentialTheft)")
    alertType: Optional[str] = Field(None, description="Alternative alert type field")
    timestamp: Optional[str] = Field(None, description="Event timestamp")
    created_timestamp: Optional[str] = Field(None, description="Alternative timestamp field")

    # User information
    user: Optional[Dict[str, Any]] = Field(None, description="User object from Falcon")
    user_id: Optional[str] = Field(None, description="User identifier")
    username: Optional[str] = Field(None, description="Username")
    user_principal_name: Optional[str] = Field(None, description="UPN (email)")

    # Network information
    source_ip: Optional[str] = Field(None, alias="sourceIp")
    destination_ip: Optional[str] = Field(None, alias="destinationIp")

    # Alert context
    severity: Optional[str] = Field("medium", description="Alert severity")
    confidence: Optional[float] = Field(None, description="Detection confidence")
    confidenceScore: Optional[float] = Field(None, description="Alternative confidence field")

    # MITRE ATT&CK
    tactics: Optional[List[str]] = Field(default_factory=list)
    techniques: Optional[List[str]] = Field(default_factory=list)

    # Action
    action: Optional[str] = Field("detected", description="Action taken")
    indicators: Optional[List[str]] = Field(default_factory=list)

    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(None)
    incident_id: Optional[str] = Field(None)

    class Config:
        json_schema_extra = {
            "example": {
                "alertId": "falcon-alert-12345",
                "type": "CredentialTheft",
                "timestamp": "2025-12-04T10:30:00Z",
                "user": {
                    "id": "U001",
                    "name": "john.doe",
                    "userPrincipalName": "john.doe@company.com"
                },
                "sourceIp": "192.168.1.100",
                "severity": "high",
                "confidence": 0.85,
                "tactics": ["TA0006"],
                "techniques": ["T1110"],
                "action": "detected"
            }
        }


class FalconCorrelatedAlert(BaseModel):
    """
    Response schema for correlated Falcon + ML alerts.

    v1.1 Enhancement - December 2025
    Combines Falcon ITDR alerts with AI Access Sentinel ML detections.
    """
    correlation_id: str = Field(..., description="Unique correlation identifier")
    correlation_confidence: str = Field(..., description="Correlation confidence level")
    correlation_score: float = Field(..., description="Correlation score (0-100)")

    # Falcon alert details
    falcon_alert_id: Optional[str] = Field(None, description="Original Falcon alert ID")
    falcon_alert_type: Optional[str] = Field(None, description="Falcon alert type")
    falcon_severity: Optional[str] = Field(None, description="Falcon severity")

    # ML detection details
    ml_anomaly_detected: bool = Field(False, description="Whether ML detected anomaly")
    ml_risk_score: Optional[float] = Field(None, description="ML risk score")

    # Combined assessment
    combined_risk_level: str = Field(..., description="Combined risk level")
    combined_risk_score: float = Field(..., description="Combined risk score")

    # User context
    user_id: Optional[str] = Field(None)
    username: Optional[str] = Field(None)

    # Response recommendations
    recommendations: List[str] = Field(default_factory=list)
    automated_actions: List[str] = Field(default_factory=list)

    timestamp: datetime = Field(default_factory=datetime.now)


class FalconWebhookResponse(BaseModel):
    """Response schema for Falcon webhook processing."""
    status: str = Field(..., description="Processing status")
    events_processed: int = Field(..., description="Number of events processed")
    correlations_found: int = Field(0, description="Number of correlations found")
    alerts_generated: int = Field(0, description="Number of new alerts generated")
    correlated_alerts: List[FalconCorrelatedAlert] = Field(default_factory=list)
    processing_time_ms: Optional[float] = Field(None)


class FalconConnectionStatus(BaseModel):
    """Status of CrowdStrike Falcon connection."""
    connected: bool = Field(..., description="Whether connected to Falcon API")
    api_version: Optional[str] = Field(None)
    last_sync: Optional[datetime] = Field(None)
    alerts_fetched: int = Field(0)
    error_message: Optional[str] = Field(None)


class FalconEnrichedRiskScore(BaseModel):
    """
    Enhanced user risk score with Falcon threat intelligence.

    v1.1 Enhancement - December 2025
    """
    user_id: str
    risk_score: float = Field(..., description="Combined risk score (0-100)")
    risk_level: str = Field(..., description="Risk level category")

    # Factor breakdown
    factor_scores: Dict[str, float] = Field(..., description="Individual factor scores")

    # Falcon-specific context
    falcon_enabled: bool = Field(True)
    falcon_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Falcon threat context if available"
    )

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "U001",
                "risk_score": 78.5,
                "risk_level": "HIGH",
                "factor_scores": {
                    "anomaly_score": 65.0,
                    "peer_deviation": 45.0,
                    "sensitive_access": 72.0,
                    "failed_attempts": 30.0,
                    "policy_violations": 55.0,
                    "falcon_threat": 85.0
                },
                "falcon_enabled": True,
                "falcon_context": {
                    "active_alerts": 2,
                    "alert_types": ["credential_theft", "lateral_movement"],
                    "max_severity": "high"
                },
                "recommendations": [
                    "HIGH: Falcon threat intelligence indicates compromise risk",
                    "Review CrowdStrike Falcon console for detailed indicators",
                    "Require immediate security review"
                ]
            }
        }
