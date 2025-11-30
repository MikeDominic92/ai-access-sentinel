"""
FastAPI main application.

AI Access Sentinel REST API.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd
import os
from typing import Dict

from src.api.schemas import (
    AccessEventRequest, AnomalyAnalysisResponse,
    BatchAnalysisRequest, BatchAnalysisResponse,
    UserRiskScoreResponse, RoleDiscoveryResponse,
    ModelMetricsResponse, HealthResponse,
    AccessPredictionRequest, AccessPredictionResponse
)
from src.models.anomaly_detector import AnomalyDetector
from src.models.access_predictor import AccessPredictor
from src.models.role_miner import RoleMiner
from src.models.risk_scorer import RiskScorer
from src.data.preprocessors import IAMDataPreprocessor

# Initialize FastAPI app
app = FastAPI(
    title="AI Access Sentinel",
    description="ML-powered IAM anomaly detection and governance API",
    version="1.0.0",
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
        self.risk_scorer: RiskScorer = RiskScorer()
        self.preprocessor: IAMDataPreprocessor = IAMDataPreprocessor()
        self.data_df: pd.DataFrame = None
        self.models_loaded: Dict[str, bool] = {
            'anomaly_detector': False,
            'access_predictor': False,
            'role_miner': False
        }

state = AppState()


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    print("Starting AI Access Sentinel API...")

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

    print("API startup complete!")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "AI Access Sentinel API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=state.models_loaded,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
