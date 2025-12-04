"""
FastAPI main application.

AI Access Sentinel REST API.

v1.1 Enhancement - December 2025:
- Added CrowdStrike Falcon ITDR webhook endpoints
- Added Falcon alert correlation with ML detections
- Enhanced risk scoring with Falcon threat intelligence
"""

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd
import os
import time
import logging
from typing import Dict, List, Optional

from src.api.schemas import (
    AccessEventRequest, AnomalyAnalysisResponse,
    BatchAnalysisRequest, BatchAnalysisResponse,
    UserRiskScoreResponse, RoleDiscoveryResponse,
    ModelMetricsResponse, HealthResponse,
    AccessPredictionRequest, AccessPredictionResponse,
    # v1.1: Falcon schemas
    FalconWebhookEvent, FalconWebhookResponse,
    FalconCorrelatedAlert, FalconConnectionStatus,
    FalconEnrichedRiskScore
)
from src.models.anomaly_detector import AnomalyDetector
from src.models.access_predictor import AccessPredictor
from src.models.role_miner import RoleMiner
from src.models.risk_scorer import RiskScorer
from src.data.preprocessors import IAMDataPreprocessor

# v1.1: Falcon integration imports
from src.integrations.crowdstrike_connector import CrowdStrikeConnector, FalconConfig
from src.integrations.falcon_event_parser import FalconEventParser
from src.integrations.alert_correlator import AlertCorrelator

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Access Sentinel",
    description="ML-powered IAM anomaly detection and governance API with CrowdStrike Falcon ITDR integration",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for models and data
class AppState:
    def __init__(self):
        self.anomaly_detector: AnomalyDetector = None
        self.access_predictor: AccessPredictor = None
        self.role_miner: RoleMiner = None
        self.risk_scorer: RiskScorer = RiskScorer(enable_falcon=True)  # v1.1
        self.preprocessor: IAMDataPreprocessor = IAMDataPreprocessor()
        self.data_df: pd.DataFrame = None
        self.models_loaded: Dict[str, bool] = {
            'anomaly_detector': False,
            'access_predictor': False,
            'role_miner': False
        }

        # v1.1: CrowdStrike Falcon ITDR components
        self.falcon_connector: CrowdStrikeConnector = None
        self.falcon_parser: FalconEventParser = FalconEventParser()
        self.alert_correlator: AlertCorrelator = AlertCorrelator()
        self.falcon_alerts_cache: List[Dict] = []
        self.falcon_last_sync: Optional[datetime] = None
        self.falcon_connected: bool = False


state = AppState()


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    print("Starting AI Access Sentinel API v1.1...")

    # Load sample data
    data_path = 'data/sample_iam_logs.csv'
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        state.data_df = pd.read_csv(data_path)
        state.data_df = state.preprocessor.preprocess_for_training(state.data_df)
        print(f"Loaded {len(state.data_df)} events")
    else:
        print(f"Warning: {data_path} not found. Generate data first.")

    # Try to load pre-trained models
    model_dir = 'models/trained'

    # Anomaly detector
    anomaly_path = os.path.join(model_dir, 'anomaly_detector_if.joblib')
    if os.path.exists(anomaly_path):
        print(f"Loading anomaly detector from {anomaly_path}...")
        state.anomaly_detector = AnomalyDetector()
        state.anomaly_detector.load(anomaly_path)
        state.models_loaded['anomaly_detector'] = True
    else:
        print("Anomaly detector not found, will train on first use")

    # Access predictor
    predictor_path = os.path.join(model_dir, 'access_predictor.joblib')
    if os.path.exists(predictor_path):
        print(f"Loading access predictor from {predictor_path}...")
        state.access_predictor = AccessPredictor()
        state.access_predictor.load(predictor_path)
        state.models_loaded['access_predictor'] = True
    else:
        print("Access predictor not found, will train on first use")

    # Role miner
    miner_path = os.path.join(model_dir, 'role_miner.joblib')
    if os.path.exists(miner_path):
        print(f"Loading role miner from {miner_path}...")
        state.role_miner = RoleMiner()
        state.role_miner.load(miner_path)
        state.models_loaded['role_miner'] = True
    else:
        print("Role miner not found, will train on first use")

    # v1.1: Initialize CrowdStrike Falcon connector
    print("\n[v1.1] Initializing CrowdStrike Falcon ITDR integration...")
    falcon_client_id = os.getenv('FALCON_CLIENT_ID')
    falcon_client_secret = os.getenv('FALCON_CLIENT_SECRET')

    if falcon_client_id and falcon_client_secret:
        falcon_config = FalconConfig(
            client_id=falcon_client_id,
            client_secret=falcon_client_secret,
            base_url=os.getenv('FALCON_BASE_URL', 'https://api.crowdstrike.com')
        )
        state.falcon_connector = CrowdStrikeConnector(falcon_config)
        if state.falcon_connector.connect():
            state.falcon_connected = True
            print("[v1.1] CrowdStrike Falcon connected successfully!")
        else:
            print("[v1.1] Warning: Could not connect to CrowdStrike Falcon")
    else:
        print("[v1.1] Falcon credentials not configured - using mock mode for demos")
        # Initialize with mock mode for demonstration
        state.falcon_connector = CrowdStrikeConnector(
            FalconConfig(client_id='demo', client_secret='demo', mock_mode=True)
        )
        state.falcon_connected = True  # Mock mode always "connected"

    print("\nAPI startup complete! v1.1 with Falcon ITDR ready.")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "AI Access Sentinel API",
        "version": "1.1.0",
        "docs": "/docs",
        "health": "/health",
        "features": {
            "ml_anomaly_detection": True,
            "risk_scoring": True,
            "role_mining": True,
            "falcon_itdr": state.falcon_connected  # v1.1
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    # v1.1: Include Falcon status in health check
    models_with_falcon = {
        **state.models_loaded,
        'falcon_connector': state.falcon_connected
    }

    return HealthResponse(
        status="healthy",
        version="1.1.0",
        models_loaded=models_with_falcon,
        timestamp=datetime.now()
    )


@app.post("/api/v1/analyze/access", response_model=AnomalyAnalysisResponse, tags=["Analysis"])
async def analyze_access_event(event: AccessEventRequest):
    """
    Analyze a single access event for anomalies.

    Returns anomaly detection results with risk scoring.
    """
    if not state.models_loaded['anomaly_detector']:
        if state.data_df is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No training data available"
            )

        # Train model on first use
        print("Training anomaly detector...")
        state.anomaly_detector = AnomalyDetector('isolation_forest')
        feature_cols = state.preprocessor.get_feature_columns()
        X = state.data_df[feature_cols].values
        state.anomaly_detector.train(X, feature_names=feature_cols)
        state.models_loaded['anomaly_detector'] = True

    # Convert event to DataFrame and preprocess
    event_df = pd.DataFrame([event.dict()])
    event_df = state.preprocessor.preprocess_for_training(event_df, include_labels=False)

    # Extract features
    feature_cols = state.preprocessor.get_feature_columns()
    features = {col: event_df[col].iloc[0] for col in feature_cols if col in event_df.columns}

    # Analyze
    result = state.anomaly_detector.analyze_event(features)

    # Generate reasons
    reasons = []
    if event_df['is_business_hours'].iloc[0] == 0:
        reasons.append("Access outside business hours")
    if event_df.get('is_suspicious_location', pd.Series([0])).iloc[0] == 1:
        reasons.append("Access from suspicious location")
    if event_df.get('combined_risk_score', pd.Series([0])).iloc[0] >= 6:
        reasons.append("High-risk resource and action combination")
    if not event.success:
        reasons.append("Failed access attempt")

    if result['is_anomaly'] and not reasons:
        reasons.append("Unusual pattern detected by ML model")

    # Recommendation
    if result['risk_level'] in ['CRITICAL', 'HIGH']:
        recommendation = "BLOCK"
    elif result['risk_level'] == 'MEDIUM':
        recommendation = "REVIEW"
    else:
        recommendation = "ALLOW"

    return AnomalyAnalysisResponse(
        is_anomaly=result['is_anomaly'],
        risk_score=result['risk_score'],
        anomaly_score=result['anomaly_score'],
        risk_level=result['risk_level'],
        reasons=reasons if reasons else ["Normal access pattern"],
        recommendation=recommendation
    )


@app.post("/api/v1/analyze/batch", response_model=BatchAnalysisResponse, tags=["Analysis"])
async def analyze_batch(request: BatchAnalysisRequest):
    """
    Analyze multiple access events in batch.

    Efficient batch processing for multiple events.
    """
    results = []
    anomaly_count = 0

    for event in request.events:
        result = await analyze_access_event(event)
        results.append(result)
        if result.is_anomaly:
            anomaly_count += 1

    return BatchAnalysisResponse(
        total_events=len(request.events),
        anomalies_detected=anomaly_count,
        anomaly_ratio=anomaly_count / len(request.events) if request.events else 0,
        results=results
    )


@app.get("/api/v1/user/{user_id}/risk-score", response_model=UserRiskScoreResponse, tags=["Risk"])
async def get_user_risk_score(user_id: str):
    """
    Get comprehensive risk score for a user.

    Analyzes multiple factors to calculate risk score.
    """
    if state.data_df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No data available"
        )

    # Check if user exists
    if user_id not in state.data_df['user_id'].values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )

    # Calculate risk score
    result = state.risk_scorer.calculate_user_risk_score(state.data_df, user_id)

    return UserRiskScoreResponse(**result)


@app.get("/api/v1/anomalies", tags=["Analysis"])
async def list_anomalies(limit: int = 100):
    """
    List detected anomalies.

    Returns recent anomalous events.
    """
    if state.data_df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No data available"
        )

    # Filter anomalies
    if 'is_anomaly' in state.data_df.columns:
        anomalies = state.data_df[state.data_df['is_anomaly'] == True].head(limit)
    else:
        anomalies = pd.DataFrame()

    return {
        "count": len(anomalies),
        "anomalies": anomalies.to_dict('records')
    }


@app.post("/api/v1/roles/discover", response_model=RoleDiscoveryResponse, tags=["Roles"])
async def discover_roles(n_clusters: int = 8):
    """
    Discover roles through clustering.

    Identifies implicit roles based on access patterns.
    """
    if state.data_df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No data available"
        )

    # Train or use existing role miner
    if not state.models_loaded['role_miner']:
        print(f"Training role miner with {n_clusters} clusters...")
        state.role_miner = RoleMiner(n_clusters=n_clusters)
        state.role_miner.train(state.data_df)
        state.models_loaded['role_miner'] = True

    # Get results
    role_summary = state.role_miner.get_role_summary()
    role_health = state.role_miner.detect_role_explosion()

    return RoleDiscoveryResponse(
        total_roles=len(role_summary),
        total_users=role_health['total_users'],
        roles=role_summary.to_dict('records'),
        role_health=role_health
    )


@app.get("/api/v1/model/metrics", response_model=ModelMetricsResponse, tags=["Models"])
async def get_model_metrics():
    """
    Get model performance metrics.

    Returns evaluation metrics for loaded models.
    """
    metrics = {
        "anomaly_detector": {
            "algorithm": "isolation_forest",
            "contamination": 0.05,
            "status": "loaded" if state.models_loaded['anomaly_detector'] else "not_loaded"
        }
    }

    if state.models_loaded['access_predictor']:
        metrics["access_predictor"] = {
            "status": "loaded",
            "model_type": "random_forest"
        }

    if state.models_loaded['role_miner']:
        role_health = state.role_miner.detect_role_explosion()
        metrics["role_miner"] = {
            "status": "loaded",
            "total_roles": role_health['total_roles'],
            "total_users": role_health['total_users']
        }

    return ModelMetricsResponse(**metrics)


@app.post("/api/v1/predict/access", response_model=AccessPredictionResponse, tags=["Prediction"])
async def predict_access(request: AccessPredictionRequest):
    """
    Predict whether access should be approved.

    Uses peer analysis and ML model.
    """
    if not state.models_loaded['access_predictor']:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Access predictor not loaded"
        )

    user_info = {
        'department': request.department,
        'job_title': request.job_title,
        'user_total_events': 100,
        'user_unique_resources': 10,
        'user_success_rate': 0.95,
        'resource_sensitivity_score': 2,
        'action_risk_score': 2,
        'combined_risk_score': 4,
        'is_business_hours': 1,
        'is_suspicious_location': 0
    }

    result = state.access_predictor.predict_access(
        user_info,
        request.resource,
        request.action
    )

    # Get peer analysis if data available
    peer_analysis = None
    if state.data_df is not None:
        peer_analysis = state.access_predictor.analyze_peer_access(
            state.data_df,
            request.department,
            request.job_title,
            request.resource
        )

    return AccessPredictionResponse(
        **result,
        peer_analysis=peer_analysis
    )


# ============================================================================
# v1.1: CrowdStrike Falcon ITDR Endpoints
# ============================================================================

@app.get("/api/v1/falcon/status", response_model=FalconConnectionStatus, tags=["Falcon ITDR"])
async def get_falcon_status():
    """
    Get CrowdStrike Falcon connection status.

    v1.1 Enhancement - December 2025
    Returns the current status of the Falcon ITDR integration.
    """
    return FalconConnectionStatus(
        connected=state.falcon_connected,
        api_version="1.1.0",
        last_sync=state.falcon_last_sync,
        alerts_fetched=len(state.falcon_alerts_cache),
        error_message=None if state.falcon_connected else "Falcon not connected"
    )


@app.post("/api/v1/falcon/webhook", response_model=FalconWebhookResponse, tags=["Falcon ITDR"])
async def receive_falcon_webhook(
    events: List[FalconWebhookEvent],
    background_tasks: BackgroundTasks
):
    """
    Receive CrowdStrike Falcon webhook events.

    v1.1 Enhancement - December 2025

    This endpoint receives identity protection alerts from CrowdStrike Falcon
    and correlates them with AI Access Sentinel ML detections.

    Supported alert types:
    - CredentialTheft
    - LateralMovement
    - PrivilegeEscalation
    - ImpossibleTravel
    - BruteForce
    - PasswordSpray
    - MFABypass
    - SessionHijack
    - GoldenTicket
    - Kerberoasting
    - DCSync
    """
    start_time = time.time()
    correlated_alerts = []
    alerts_generated = 0

    for event in events:
        # Parse the Falcon event
        normalized = state.falcon_parser.parse_identity_alert(event.dict())

        # Cache the alert
        state.falcon_alerts_cache.append(normalized.raw_event)

        # Try to correlate with ML detections
        if state.data_df is not None and normalized.user_id:
            user_events = state.data_df[
                state.data_df['user_id'] == normalized.user_id
            ]

            if len(user_events) > 0:
                # Create AI Access Sentinel alert from recent anomalies
                recent_anomalies = user_events[
                    user_events.get('is_anomaly', pd.Series([False]*len(user_events)))
                ].head(5)

                if len(recent_anomalies) > 0:
                    from src.integrations.alert_correlator import AIAccessSentinelAlert

                    # Create a synthetic ML alert for correlation
                    ml_alert = AIAccessSentinelAlert(
                        alert_id=f"ml-{normalized.user_id}-{int(time.time())}",
                        user_id=normalized.user_id,
                        alert_type="ml_anomaly_detection",
                        severity="medium",
                        timestamp=datetime.now(),
                        risk_score=float(recent_anomalies.get('risk_score', pd.Series([50])).mean()),
                        is_anomaly=True
                    )

                    # Correlate
                    correlation = state.alert_correlator.correlate_alert(
                        normalized, ml_alert
                    )

                    if correlation and correlation.correlation_score > 30:
                        correlated_alerts.append(FalconCorrelatedAlert(
                            correlation_id=correlation.correlation_id,
                            correlation_confidence=correlation.correlation_confidence.value,
                            correlation_score=correlation.correlation_score,
                            falcon_alert_id=normalized.event_id,
                            falcon_alert_type=normalized.event_type.value,
                            falcon_severity=normalized.severity,
                            ml_anomaly_detected=True,
                            ml_risk_score=ml_alert.risk_score,
                            combined_risk_level=correlation.combined_risk_level,
                            combined_risk_score=correlation.combined_risk_score,
                            user_id=normalized.user_id,
                            username=normalized.username,
                            recommendations=correlation.recommendations,
                            automated_actions=correlation.automated_actions,
                            timestamp=datetime.now()
                        ))
                        alerts_generated += 1

    # Update last sync time
    state.falcon_last_sync = datetime.now()

    processing_time = (time.time() - start_time) * 1000

    return FalconWebhookResponse(
        status="processed",
        events_processed=len(events),
        correlations_found=len(correlated_alerts),
        alerts_generated=alerts_generated,
        correlated_alerts=correlated_alerts,
        processing_time_ms=round(processing_time, 2)
    )


@app.get(
    "/api/v1/falcon/alerts",
    tags=["Falcon ITDR"]
)
async def get_falcon_alerts(
    limit: int = 100,
    severity: Optional[str] = None
):
    """
    Get cached Falcon alerts.

    v1.1 Enhancement - December 2025
    Returns recent Falcon ITDR alerts from the cache.
    """
    alerts = state.falcon_alerts_cache[-limit:]

    if severity:
        alerts = [a for a in alerts if a.get('severity', '').lower() == severity.lower()]

    return {
        "count": len(alerts),
        "alerts": alerts,
        "last_sync": state.falcon_last_sync
    }


@app.get(
    "/api/v1/falcon/user/{user_id}/risk",
    response_model=FalconEnrichedRiskScore,
    tags=["Falcon ITDR"]
)
async def get_falcon_enriched_risk_score(user_id: str):
    """
    Get user risk score with Falcon threat intelligence.

    v1.1 Enhancement - December 2025

    Calculates comprehensive risk score that includes:
    - Traditional ML-based factors (anomaly, peer deviation, etc.)
    - CrowdStrike Falcon threat intelligence factor (25% weight)
    - Active Falcon alerts for the user
    - Threat indicators from Falcon
    """
    if state.data_df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No data available"
        )

    if user_id not in state.data_df['user_id'].values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )

    # Get Falcon enrichment if available
    falcon_enrichment = None
    if state.falcon_connector and state.falcon_connected:
        # Check for user-specific Falcon alerts
        user_alerts = [
            a for a in state.falcon_alerts_cache
            if a.get('user_id') == user_id or
               a.get('user_principal_name', '').lower() == user_id.lower()
        ]

        if user_alerts:
            falcon_enrichment = {
                'has_falcon_alerts': True,
                'active_alerts': user_alerts,
                'falcon_risk_boost': len(user_alerts) * 5,
                'threat_indicators': []
            }

    # Calculate risk score with Falcon data
    result = state.risk_scorer.calculate_user_risk_score(
        state.data_df,
        user_id,
        falcon_alerts=state.falcon_alerts_cache,
        falcon_enrichment=falcon_enrichment
    )

    return FalconEnrichedRiskScore(**result)


@app.post("/api/v1/falcon/sync", tags=["Falcon ITDR"])
async def sync_falcon_alerts(background_tasks: BackgroundTasks):
    """
    Manually trigger sync with CrowdStrike Falcon.

    v1.1 Enhancement - December 2025
    Fetches recent identity protection alerts from Falcon API.
    """
    if not state.falcon_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Falcon connector not available"
        )

    try:
        # Fetch alerts from Falcon
        alerts = state.falcon_connector.get_identity_alerts(limit=100)

        # Parse and cache
        for alert in alerts:
            normalized = state.falcon_parser.parse_identity_alert(alert)
            state.falcon_alerts_cache.append(normalized.raw_event)

        state.falcon_last_sync = datetime.now()

        return {
            "status": "synced",
            "alerts_fetched": len(alerts),
            "total_cached": len(state.falcon_alerts_cache),
            "last_sync": state.falcon_last_sync
        }

    except Exception as e:
        logger.error(f"Falcon sync failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falcon sync failed: {str(e)}"
        )


@app.get("/api/v1/falcon/correlations", tags=["Falcon ITDR"])
async def get_correlations(limit: int = 50):
    """
    Get recent alert correlations.

    v1.1 Enhancement - December 2025
    Returns correlations between Falcon ITDR and ML detections.
    """
    correlations = state.alert_correlator.get_recent_correlations(limit=limit)

    return {
        "count": len(correlations),
        "correlations": [
            {
                "correlation_id": c.correlation_id,
                "correlation_confidence": c.correlation_confidence.value,
                "correlation_score": c.correlation_score,
                "user_id": c.user_id,
                "combined_risk_level": c.combined_risk_level,
                "timestamp": c.timestamp.isoformat() if c.timestamp else None
            }
            for c in correlations
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
