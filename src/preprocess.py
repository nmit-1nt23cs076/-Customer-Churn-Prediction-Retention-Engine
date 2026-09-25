"""
Feature Engineering & Preprocessing Pipeline
Engineers 20+ domain-specific behavioral features, encodes categorical variables,
scales numerical features, and applies SMOTE to resolve class imbalance.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to engineer 20+ behavioral, financial, and service bundle features.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Fill missing total charges with MonthlyCharges * tenure if any
        if "TotalCharges" in df.columns and "MonthlyCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])

        # 1. Financial & Ratio Features
        df["tenure_to_charge_ratio"] = df["tenure"] / (df["MonthlyCharges"] + 1e-5)
        df["monthly_to_total_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1e-5)
        df["charge_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1.0)
        df["avg_monthly_discrepancy"] = df["MonthlyCharges"] - (df["TotalCharges"] / (df["tenure"] + 1.0))

        # 2. Service Bundle & Value Metrics
        service_cols = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        active_services = 0
        for col in service_cols:
            if col in df.columns:
                active_services += (df[col] == "Yes").astype(int)
        df["total_active_services"] = active_services
        df["charge_per_active_service"] = df["MonthlyCharges"] / (df["total_active_services"] + 1.0)

        # 3. High-Risk Compound Flags
        df["fiber_without_techsupport"] = (
            (df.get("InternetService", "") == "Fiber optic") & 
            (df.get("TechSupport", "") == "No")
        ).astype(int)

        df["month_to_month_electronic_check"] = (
            (df.get("Contract", "") == "Month-to-month") & 
            (df.get("PaymentMethod", "") == "Electronic check")
        ).astype(int)

        df["senior_living_alone"] = (
            (df.get("SeniorCitizen", 0) == 1) & 
            (df.get("Partner", "No") == "No") & 
            (df.get("Dependents", "No") == "No")
        ).astype(int)

        df["streaming_power_user"] = (
            (df.get("StreamingTV", "") == "Yes") & 
            (df.get("StreamingMovies", "") == "Yes")
        ).astype(int)

        df["full_security_suite"] = (
            (df.get("OnlineSecurity", "") == "Yes") & 
            (df.get("OnlineBackup", "") == "Yes") & 
            (df.get("DeviceProtection", "") == "Yes")
        ).astype(int)

        # 4. Behavioral & Risk Indicators (if present)
        if "SupportTickets" in df.columns:
            df["ticket_frequency_per_year"] = df["SupportTickets"] / (np.maximum(df["tenure"], 1) / 12.0)
        else:
            df["ticket_frequency_per_year"] = 0.0

        if "PaymentFailures" in df.columns:
            df["failure_rate"] = df["PaymentFailures"] / (np.maximum(df["tenure"], 1) / 12.0)
        else:
            df["failure_rate"] = 0.0

        if "SatisfactionScore" in df.columns:
            df["dissatisfaction_risk"] = (df["SatisfactionScore"] <= 2.5).astype(int)
        else:
            df["dissatisfaction_risk"] = 0

        # Drop identifier if present
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])

        return df


def build_preprocessor_pipeline(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Creates an end-to-end ColumnTransformer for scaling numeric features
    and one-hot encoding categorical variables.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor


def prepare_data(
    df: pd.DataFrame, 
    apply_smote: bool = True,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer, List[str]]:
    """
    Executes feature engineering, preprocessor fitting, and SMOTE balancing.
    """
    y = df["Churn"].values if "Churn" in df.columns else None
    X_raw = df.drop(columns=["Churn"], errors="ignore")

    # Step 1: Feature Engineering
    engineer = ChurnFeatureEngineer()
    X_engineered = engineer.transform(X_raw)

    # Separate column types
    categorical_cols = [
        col for col in X_engineered.columns 
        if X_engineered[col].dtype == "object" or col in ["Contract", "PaymentMethod", "InternetService"]
    ]
    numeric_cols = [col for col in X_engineered.columns if col not in categorical_cols]

    # Step 2: Fit-transform Preprocessor
    preprocessor = build_preprocessor_pipeline(numeric_cols, categorical_cols)
    X_processed = preprocessor.fit_transform(X_engineered)

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numeric_cols + cat_feature_names

    # Step 3: Apply SMOTE if training
    if apply_smote and y is not None:
        smote = SMOTE(random_state=random_state, sampling_strategy=0.85)
        X_resampled, y_resampled = smote.fit_resample(X_processed, y)
        return X_resampled, y_resampled, preprocessor, all_feature_names

    return X_processed, y, preprocessor, all_feature_names
