"""
Dataset Generator: 25,000+ Enterprise Customer Records
Generates realistic multi-feature telecom/subscription customer data
with realistic churn indicators (tenure, pricing, contract types, service bundles).
"""

import os
import numpy as np
import pandas as pd

def generate_customer_churn_dataset(num_records: int = 25000, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    
    # 1. Customer Demographics
    customer_ids = [f"CUST-{100000 + i}" for i in range(num_records)]
    senior_citizen = np.random.choice([0, 1], size=num_records, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], size=num_records, p=[0.48, 0.52])
    dependents = np.where(partner == "Yes", np.random.choice(["Yes", "No"], size=num_records, p=[0.45, 0.55]), "No")
    
    # 2. Account Tenure (Exponential decay + long-term retention distribution)
    tenure_months = np.clip(np.random.gamma(shape=2.0, scale=14.0, size=num_records).astype(int), 1, 72)
    
    # 3. Contract & Payment Preferences
    # Newer customers have higher probability of Month-to-Month
    contract = np.empty(num_records, dtype=object)
    short_tenure_mask = (tenure_months < 12)
    n_short = np.sum(short_tenure_mask)
    n_long = num_records - n_short
    contract[short_tenure_mask] = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_short, p=[0.78, 0.15, 0.07])
    contract[~short_tenure_mask] = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_long, p=[0.35, 0.35, 0.30])
    
    paperless_billing = np.random.choice(["Yes", "No"], size=num_records, p=[0.60, 0.40])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=num_records,
        p=[0.34, 0.18, 0.24, 0.24]
    )
    
    # 4. Subscribed Services
    internet_service = np.random.choice(["Fiber optic", "DSL", "No"], size=num_records, p=[0.45, 0.35, 0.20])
    online_security = np.where(internet_service == "No", "No internet service", np.random.choice(["Yes", "No"], size=num_records, p=[0.38, 0.62]))
    tech_support = np.where(internet_service == "No", "No internet service", np.random.choice(["Yes", "No"], size=num_records, p=[0.35, 0.65]))
    online_backup = np.where(internet_service == "No", "No internet service", np.random.choice(["Yes", "No"], size=num_records, p=[0.40, 0.60]))
    device_protection = np.where(internet_service == "No", "No internet service", np.random.choice(["Yes", "No"], size=num_records, p=[0.39, 0.61]))
    streaming_tv = np.where(internet_service == "No", "No internet service", np.random.choice(["Yes", "No"], size=num_records, p=[0.42, 0.58]))
    streaming_movies = np.where(internet_service == "No", "No internet service", np.random.choice(["Yes", "No"], size=num_records, p=[0.43, 0.57]))
    multiple_lines = np.random.choice(["Yes", "No", "No phone service"], size=num_records, p=[0.42, 0.48, 0.10])
    
    # 5. Billing Charges & Usage Signals
    # Base charges depend heavily on internet service and add-ons
    base_charge = np.where(internet_service == "Fiber optic", 75.0, np.where(internet_service == "DSL", 45.0, 20.0))
    addon_cost = (
        (online_security == "Yes") * 8.0 +
        (tech_support == "Yes") * 8.0 +
        (online_backup == "Yes") * 6.0 +
        (device_protection == "Yes") * 6.0 +
        (streaming_tv == "Yes") * 10.0 +
        (streaming_movies == "Yes") * 10.0 +
        (multiple_lines == "Yes") * 12.0
    )
    monthly_charges = np.round(base_charge + addon_cost + np.random.normal(0, 3.5, size=num_records), 2)
    monthly_charges = np.clip(monthly_charges, 18.50, 125.00)
    
    # Total charges with slight variation for missed/late payments
    total_charges = np.round(monthly_charges * tenure_months + np.random.normal(0, 15, size=num_records), 2)
    total_charges = np.clip(total_charges, 18.50, None)
    
    # Additional Enterprise Behavioral Features
    support_tickets = np.random.poisson(lam=np.where(tech_support == "No", 2.2, 0.8), size=num_records)
    payment_failures_last_year = np.random.binomial(n=4, p=np.where(payment_method == "Electronic check", 0.35, 0.08), size=num_records)
    satisfaction_score = np.clip(np.random.normal(loc=np.where(contract == "Month-to-month", 3.2, 4.4), scale=1.0), 1, 5).round(1)

    # 6. Realistic Churn Probability Generation (Latent Log-Odds)
    log_odds = (
        -2.2
        + (contract == "Month-to-month") * 1.6
        - (contract == "Two year") * 1.4
        + (internet_service == "Fiber optic") * 0.8
        - (tech_support == "Yes") * 0.7
        - (online_security == "Yes") * 0.6
        + (payment_method == "Electronic check") * 0.65
        + (payment_failures_last_year * 0.45)
        + (support_tickets * 0.25)
        - (tenure_months / 18.0)
        + (monthly_charges / 85.0)
        - (satisfaction_score * 0.5)
        + (paperless_billing == "Yes") * 0.25
        + np.random.normal(0, 0.4, size=num_records)
    )
    
    churn_prob = 1.0 / (1.0 + np.exp(-log_odds))
    churn = (np.random.rand(num_records) < churn_prob).astype(int)
    
    df = pd.DataFrame({
        "customerID": customer_ids,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure_months,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "MultipleLines": multiple_lines,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SupportTickets": support_tickets,
        "PaymentFailures": payment_failures_last_year,
        "SatisfactionScore": satisfaction_score,
        "Churn": churn
    })
    
    return df

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(out_dir, "customer_churn_25k.csv")
    print(f"Generating 25,000 customer records...")
    df = generate_customer_churn_dataset(num_records=25000)
    df.to_csv(data_path, index=False)
    print(f"Successfully saved 25,000 records to: {data_path}")
    print(f"Class Distribution: {df['Churn'].value_counts(normalize=True).to_dict()}")
