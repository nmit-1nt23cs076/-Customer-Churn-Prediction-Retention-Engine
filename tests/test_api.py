"""
Integration tests for FastAPI endpoints
"""

import sys
import os
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from api.main import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "algorithm" in data


def test_predict_single_endpoint():
    payload = {
        "customerID": "TEST-101",
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "tenure": 3,
        "MonthlyCharges": 95.0,
        "PaymentMethod": "Electronic check",
        "TechSupport": "No"
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customerID"] == "TEST-101"
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert "inference_time_ms" in data


def test_explain_endpoint():
    payload = {
        "customerID": "TEST-102",
        "Contract": "Two year",
        "InternetService": "DSL",
        "tenure": 36,
        "MonthlyCharges": 45.0,
        "PaymentMethod": "Credit card (automatic)",
        "TechSupport": "Yes"
    }
    response = client.post("/api/v1/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "top_risk_drivers" in data
    assert "top_retention_factors" in data


def test_batch_predict_endpoint():
    payload = {
        "customers": [
            {"customerID": "BATCH-1", "Contract": "Month-to-month", "tenure": 2, "MonthlyCharges": 80.0},
            {"customerID": "BATCH-2", "Contract": "Two year", "tenure": 60, "MonthlyCharges": 30.0}
        ]
    }
    response = client.post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 2
    assert len(data["results"]) == 2
