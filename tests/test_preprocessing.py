"""
Unit tests for data preprocessing and feature engineering
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocess import ChurnFeatureEngineer


def test_feature_engineering_creates_ratios():
    raw_data = {
        "customerID": ["CUST-1"],
        "SeniorCitizen": [0],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "tenure": [10],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "InternetService": ["Fiber optic"],
        "OnlineSecurity": ["No"],
        "TechSupport": ["No"],
        "OnlineBackup": ["Yes"],
        "DeviceProtection": ["No"],
        "StreamingTV": ["Yes"],
        "StreamingMovies": ["Yes"],
        "MultipleLines": ["Yes"],
        "MonthlyCharges": [90.0],
        "TotalCharges": [900.0],
        "SupportTickets": [3],
        "PaymentFailures": [1],
        "SatisfactionScore": [2.5]
    }
    df = pd.DataFrame(raw_data)
    engineer = ChurnFeatureEngineer()
    df_eng = engineer.transform(df)

    # Assert new ratio columns exist
    assert "tenure_to_charge_ratio" in df_eng.columns
    assert "monthly_to_total_ratio" in df_eng.columns
    assert "fiber_without_techsupport" in df_eng.columns
    assert "month_to_month_electronic_check" in df_eng.columns
    assert "total_active_services" in df_eng.columns

    # Verify compound risk flag logic
    assert df_eng["fiber_without_techsupport"].iloc[0] == 1
    assert df_eng["month_to_month_electronic_check"].iloc[0] == 1
    assert df_eng["total_active_services"].iloc[0] == 3  # Backup, StreamingTV, StreamingMovies
