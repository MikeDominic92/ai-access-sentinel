"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


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
    roles: List[Dict[str, any]]
    role_health: Dict[str, any]


class ModelMetricsResponse(BaseModel):
    """Response schema for model metrics."""
    anomaly_detector: Dict[str, float]
    access_predictor: Optional[Dict[str, float]] = None
    role_miner: Optional[Dict[str, any]] = None


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
    peer_analysis: Optional[Dict[str, any]] = None
