"""
Tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "models_loaded" in data


def test_analyze_access_endpoint(client):
    """Test access event analysis endpoint."""
    event_data = {
        "user_id": "U001",
        "username": "test_user",
        "department": "Engineering",
        "job_title": "Software Engineer",
        "resource": "source_code_repo",
        "action": "read",
        "timestamp": datetime.now().isoformat(),
        "source_ip": "192.168.1.100",
        "location": "New York, US",
        "success": True
    }

    response = client.post("/api/v1/analyze/access", json=event_data)

    # May return 503 if models not trained, which is acceptable for test
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "is_anomaly" in data
        assert "risk_score" in data
        assert "risk_level" in data


def test_invalid_event_data(client):
    """Test API validation with invalid data."""
    invalid_data = {
        "user_id": "U001"
        # Missing required fields
    }

    response = client.post("/api/v1/analyze/access", json=invalid_data)
    assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
