"""
Model Training & Benchmarking Pipeline
Benchmarks LightGBM vs. XGBoost, executes Optuna Bayesian optimization,
evaluates ROC-AUC and Recall, and saves the production model and preprocessor.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, 
    f1_score, accuracy_score, classification_report, confusion_matrix
)
import lightgbm as lgb
import xgboost as xgb

from preprocess import ChurnFeatureEngineer, prepare_data

# Suppress Optuna logging verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_optuna_tuning_lgb(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 5) -> dict:
    """
    Bayesian Hyperparameter Optimization for LightGBM using Optuna.
    """
    print(f"\n[Optuna] Starting Bayesian Hyperparameter Optimization ({n_trials} trials)...")

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": 42,
            "n_estimators": trial.suggest_int("n_estimators", 100, 350),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, y_train)
        preds_proba = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds_proba)
        return auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    print(f"[Optuna] Best Trial Validation ROC-AUC: {study.best_value:.4f}")
    return study.best_params


def train_and_benchmark(data_path: str = None, models_dir: str = None, use_optuna: bool = True):
    """
    Main training function: loads data, benchmarks LightGBM vs XGBoost,
    optimizes best model, and saves artifacts.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_path is None:
        data_path = os.path.join(base_dir, "data", "customer_churn_25k.csv")
    if models_dir is None:
        models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load or Generate Dataset
    if not os.path.exists(data_path):
        print(f"[Data] Generating fresh 25,000 customer records...")
        from data.generate_dataset import generate_customer_churn_dataset
        df = generate_customer_churn_dataset(num_records=25000)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        print(f"[Data] Loading dataset from: {data_path}")
        df = pd.read_csv(data_path)

    print(f"[Data] Total records: {len(df):,} | Churn rate: {df['Churn'].mean():.2%}")

    # 2. Train-Val-Test Split (70% Train, 15% Val, 15% Test)
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["Churn"])
    train_df, val_df = train_test_split(train_df, test_size=0.1765, random_state=42, stratify=train_df["Churn"])

    print(f"[Split] Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # 3. Fit Preprocessing & SMOTE on Train Set
    y_train = train_df["Churn"].values
    y_val = val_df["Churn"].values
    y_test = test_df["Churn"].values

    engineer = ChurnFeatureEngineer()
    X_train_eng = engineer.transform(train_df.drop(columns=["Churn"]))
    X_val_eng = engineer.transform(val_df.drop(columns=["Churn"]))
    X_test_eng = engineer.transform(test_df.drop(columns=["Churn"]))

    # Categorical & Numerical feature definition (robust check)
    num_cols = [c for c in X_train_eng.columns if pd.api.types.is_numeric_dtype(X_train_eng[c])]
    cat_cols = [c for c in X_train_eng.columns if c not in num_cols]

    from preprocess import build_preprocessor_pipeline
    preprocessor = build_preprocessor_pipeline(num_cols, cat_cols)
    X_train_proc = preprocessor.fit_transform(X_train_eng)
    X_val_proc = preprocessor.transform(X_val_eng)
    X_test_proc = preprocessor.transform(X_test_eng)

    # SMOTE balancing
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42, sampling_strategy=0.85)
    X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)
    print(f"[SMOTE] Balanced training set from {len(y_train):,} to {len(y_train_res):,} samples")

    # 4. Benchmarking: LightGBM vs XGBoost
    print("\n" + "="*50)
    print("BENCHMARKING: LightGBM vs. XGBoost")
    print("="*50)

    # LightGBM Baseline
    t0 = time.time()
    lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    lgb_model.fit(X_train_res, y_train_res)
    lgb_train_time = time.time() - t0
    lgb_preds_val = lgb_model.predict_proba(X_val_proc)[:, 1]
    lgb_auc = roc_auc_score(y_val, lgb_preds_val)

    # XGBoost Baseline
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0)
    xgb_model.fit(X_train_res, y_train_res)
    xgb_train_time = time.time() - t0
    xgb_preds_val = xgb_model.predict_proba(X_val_proc)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_preds_val)

    print(f"• LightGBM Base -> Val ROC-AUC: {lgb_auc:.4f} | Training Time: {lgb_train_time:.2f}s")
    print(f"• XGBoost  Base -> Val ROC-AUC: {xgb_auc:.4f} | Training Time: {xgb_train_time:.2f}s")

    # 5. Hyperparameter Optimization via Optuna for Winning Model
    best_params = {}
    if use_optuna:
        best_params = run_optuna_tuning_lgb(X_train_res, y_train_res, X_val_proc, y_val, n_trials=5)
        final_model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
    else:
        final_model = lgb_model

    final_model.fit(X_train_res, y_train_res)

    # 6. Evaluation on Held-out Test Set
    test_probs = final_model.predict_proba(X_test_proc)[:, 1]
    test_preds = (test_probs >= 0.45).astype(int)  # 0.45 threshold to prioritize Recall on churn

    test_auc = roc_auc_score(y_test, test_probs)
    test_recall = recall_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    test_acc = accuracy_score(y_test, test_preds)

    print("\n" + "="*50)
    print("FINAL TEST SET PERFORMANCE (Production LightGBM Model)")
    print("="*50)
    print(f"• ROC-AUC Score : {test_auc:.4f} (Resume Target: >0.88)")
    print(f"• Recall (Churn): {test_recall:.2%} (Resume Target: >83%)")
    print(f"• Precision     : {test_precision:.2%}")
    print(f"• F1-Score      : {test_f1:.4f}")
    print(f"• Accuracy      : {test_acc:.2%}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_preds))

    # 7. Save Artifacts for API Serving
    model_path = os.path.join(models_dir, "model.joblib")
    preproc_path = os.path.join(models_dir, "preprocessor.joblib")
    metrics_path = os.path.join(models_dir, "metrics.json")

    joblib.dump(final_model, model_path)
    joblib.dump(preprocessor, preproc_path)

    metrics = {
        "roc_auc": round(float(test_auc), 4),
        "recall": round(float(test_recall), 4),
        "precision": round(float(test_precision), 4),
        "f1": round(float(test_f1), 4),
        "accuracy": round(float(test_acc), 4),
        "benchmark": {
            "lightgbm_train_time_sec": round(lgb_train_time, 2),
            "xgboost_train_time_sec": round(xgb_train_time, 2),
            "lightgbm_val_auc": round(lgb_auc, 4),
            "xgboost_val_auc": round(xgb_auc, 4)
        }
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save a sample customer record for API testing
    sample_record = test_df.iloc[0].drop(labels=["Churn"]).to_dict()
    sample_path = os.path.join(base_dir, "data", "sample_customer.json")
    with open(sample_path, "w") as f:
        json.dump(sample_record, f, indent=2)

    print(f"\n[Saved] Model saved to: {model_path}")
    print(f"[Saved] Preprocessor saved to: {preproc_path}")
    print(f"[Saved] Metrics saved to: {metrics_path}")
    print(f"[Saved] Sample customer saved to: {sample_path}")

    return final_model, preprocessor, metrics


if __name__ == "__main__":
    train_and_benchmark(use_optuna=True)
