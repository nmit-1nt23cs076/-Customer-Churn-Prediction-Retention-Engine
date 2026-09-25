"""
Model Explainability Layer (SHAP)
Uses TreeExplainer to compute SHAP values for global feature importance
and real-time local customer risk factor breakdowns.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import shap
from preprocess import ChurnFeatureEngineer


class ChurnExplainer:
    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "model.joblib")
        if preprocessor_path is None:
            preprocessor_path = os.path.join(base_dir, "models", "preprocessor.joblib")

        self.model = joblib.load(model_path) if os.path.exists(model_path) else None
        self.preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
        self.engineer = ChurnFeatureEngineer()
        self.explainer = None

        if self.model is not None:
            # TreeExplainer is ultra-fast for LightGBM / XGBoost
            self.explainer = shap.TreeExplainer(self.model)

    def _get_feature_names(self) -> List[str]:
        if self.preprocessor is None:
            return []
        cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_features = cat_encoder.get_feature_names_out().tolist()
        num_features = self.preprocessor.transformers_[0][2]
        return list(num_features) + list(cat_features)

    def explain_single_customer(self, customer_dict: Dict[str, Any], top_n: int = 5) -> Dict[str, Any]:
        """
        Calculates local SHAP values for a single customer to identify top risk drivers.
        """
        if self.model is None or self.preprocessor is None:
            # Fallback heuristic explanation if model artifacts are not yet built
            return self._heuristic_fallback(customer_dict)

        df_single = pd.DataFrame([customer_dict])
        df_eng = self.engineer.transform(df_single)
        X_proc = self.preprocessor.transform(df_eng)
        feature_names = self._get_feature_names()

        # Compute SHAP values
        shap_values = self.explainer.shap_values(X_proc)
        # For binary classification in LightGBM, shap_values is a list [class_0, class_1] or single array
        if isinstance(shap_values, list) and len(shap_values) == 2:
            vals = shap_values[1][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            vals = shap_values[0, :, 1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            vals = shap_values[0]
        else:
            vals = shap_values

        # Pair feature names with SHAP contributions
        contributions = []
        for name, val in zip(feature_names, vals):
            contributions.append({
                "feature": self._clean_feature_name(name),
                "raw_name": name,
                "impact": round(float(val), 4),
                "direction": "INCREASES_CHURN" if val > 0 else "DECREASES_CHURN"
            })

        # Sort by absolute impact
        contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
        top_risk_drivers = [c for c in contributions if c["direction"] == "INCREASES_CHURN"][:top_n]
        top_retention_factors = [c for c in contributions if c["direction"] == "DECREASES_CHURN"][:top_n]

        base_val = getattr(self.explainer, "expected_value", 0.0)
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[-1])

        return {
            "base_value": round(float(base_val), 4),
            "top_risk_drivers": top_risk_drivers,
            "top_retention_factors": top_retention_factors
        }

    def _clean_feature_name(self, name: str) -> str:
        name = name.replace("onehot__", "").replace("cat__", "").replace("num__", "")
        name = name.replace("_", " ").title()
        return name

    def _heuristic_fallback(self, customer: Dict[str, Any]) -> Dict[str, Any]:
        drivers = []
        if customer.get("Contract") == "Month-to-month":
            drivers.append({"feature": "Contract: Month-to-Month", "impact": 0.42, "direction": "INCREASES_CHURN"})
        if customer.get("PaymentMethod") == "Electronic check":
            drivers.append({"feature": "Payment: Electronic Check", "impact": 0.28, "direction": "INCREASES_CHURN"})
        if customer.get("TechSupport") == "No":
            drivers.append({"feature": "No Tech Support", "impact": 0.22, "direction": "INCREASES_CHURN"})
        if customer.get("MonthlyCharges", 0) > 75:
            drivers.append({"feature": "High Monthly Charges", "impact": 0.19, "direction": "INCREASES_CHURN"})
        return {
            "base_value": 0.25,
            "top_risk_drivers": drivers,
            "top_retention_factors": [
                {"feature": "Long Tenure" if customer.get("tenure", 0) > 24 else "Paperless Billing", "impact": -0.31, "direction": "DECREASES_CHURN"}
            ]
        }
