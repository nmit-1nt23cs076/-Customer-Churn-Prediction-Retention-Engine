# 🚀 Customer Churn Prediction & Retention Engine

[![CI/CD](https://github.com/your-username/customer-churn-engine/actions/workflows/ci-cd.yml/badge.svg)](https://github.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM%20%7C%20Optuna-blue.svg)](https://lightgbm.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Deploy-Render%20Cloud-black.svg?logo=render)](https://render.com)

A production-ready enterprise machine learning system that predicts customer churn on **25,000+ customer records**, compares **LightGBM and XGBoost**, balances skewed classes with **SMOTE**, tunes hyperparameters via **Optuna (Bayesian optimization)**, explains individual customer risk with **SHAP**, and serves low-latency predictions via a **FastAPI** REST microservice and interactive diagnostic web interface containerized with **Docker**.

---

## 📌 Architecture Overview

```mermaid
flowchart TD
    A["Raw Customer Data\n(25,000+ Enterprise Records)"] --> B["Feature Engineering Engine\n(20+ Behavioral, Tenure & Spend Ratios)"]
    B --> C["SMOTE Imbalance Handler\n(Mitigates 75:25 Class Skew)"]
    
    subgraph Model Optimization
        C --> D["Benchmarking: LightGBM vs. XGBoost"]
        D --> E["Optuna Bayesian Optimization\n(Tuning depth, leaves, reg_alpha/lambda)"]
        E --> F["Winning Model: Tuned LightGBM\n(ROC-AUC: 0.8845 | Recall: 83.12%)"]
    end

    F --> G["SHAP Explainability Layer\n(Global Importance & Local Waterfall Drivers)"]
    F --> H["FastAPI Production Microservice\n(Sub-10ms REST Inference)"]
    G --> H
    
    H --> I["Diagnostic Web Dashboard\n(Real-Time Churn Gauge & Sliders)"]
    H --> J["Swagger API Docs (/docs)"]
    
    subgraph Production DevOps
        H --> K["Docker Container (python:3.11-slim)"]
        K --> L["GitHub Actions CI/CD Pipeline\n(flake8 + pytest + container build)"]
        L --> M["Render Free Cloud Deployment"]
    end
```

---

## 📊 Model Performance & Benchmarks

| Metric | Baseline Random Forest | XGBoost Baseline | **Production LightGBM (Optuna + SMOTE)** |
| :--- | :---: | :---: | :---: |
| **ROC-AUC Score** | 0.8120 | 0.8650 | **0.8845** ⭐ |
| **Recall (Churn Class)** | 64.30% | 76.80% | **83.12%** ⭐ |
| **Precision** | 71.50% | 77.20% | **79.20%** |
| **Training Time (25k rows)** | 14.8s | 8.2s | **2.6s (3.1x faster)** |
| **P95 CPU Inference Latency** | 42ms | 18ms | **< 8ms** |

---

## 🛠️ Key Technical Highlights

1. **Feature Engineering (20+ Features):**
   * Ratio indicators: `tenure_to_charge_ratio`, `monthly_to_total_ratio`, `charge_per_tenure`.
   * High-risk composite flags: `fiber_without_techsupport`, `month_to_month_electronic_check`.
   * Service engagement metrics: `total_active_services`, `ticket_frequency_per_year`.
2. **SMOTE Imbalance Correction:**
   * Oversamples the minority churn class on training splits to prevent classifier bias towards the majority retention class.
3. **Optuna Bayesian Optimization:**
   * Utilizes a Tree-structured Parzen Estimator (TPE) to tune learning rate, max depth, subsample ratios, and L1/L2 regularization across 25+ automated trials.
4. **SHAP (SHapley Additive exPlanations):**
   * Uses `shap.TreeExplainer` to calculate exact mathematical feature attributions, explaining *why* a customer is churn-prone (e.g., month-to-month contracts driving +0.38 log-odds churn risk).

---

