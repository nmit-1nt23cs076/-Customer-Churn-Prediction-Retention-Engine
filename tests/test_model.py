"""
Unit tests for model inference bounds and reproducibility
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocess import ChurnFeatureEngineer


def test_heuristic_risk_bounds():
    from api.main import compute_prediction

    # Risky customer test
    risky_customer = {
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "TechSupport": "No",
        "tenure": 2,
        "MonthlyCharges": 95.0
    }
    res_risky = compute_prediction(risky_customer)
    assert 0.0 <= res_risky["churn_probability"] <= 1.0
    assert res_risky["risk_level"] in ["MEDIUM", "HIGH"]

    # Loyal customer test
    loyal_customer = {
        "Contract": "Two year",
        "PaymentMethod": "Bank transfer (automatic)",
        "TechSupport": "Yes",
        "tenure": 48,
        "MonthlyCharges": 45.0
    }
    res_loyal = compute_prediction(loyal_customer)
    assert 0.0 <= res_loyal["churn_probability"] <= 1.0
    assert res_loyal["churn_probability"] < res_risky["churn_probability"]
