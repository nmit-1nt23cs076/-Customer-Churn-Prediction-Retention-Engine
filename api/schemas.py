"""
Pydantic Schemas for Request & Response Validation
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    customerID: Optional[str] = Field(default="CUST-PREVIEW", description="Unique customer identifier")
    SeniorCitizen: int = Field(default=0, ge=0, le=1, description="1 if senior citizen, else 0")
    Partner: str = Field(default="No", description="'Yes' or 'No'")
    Dependents: str = Field(default="No", description="'Yes' or 'No'")
    tenure: int = Field(default=12, ge=1, le=72, description="Months the customer has been with the company")
    Contract: str = Field(default="Month-to-month", description="'Month-to-month', 'One year', or 'Two year'")
    PaperlessBilling: str = Field(default="Yes", description="'Yes' or 'No'")
    PaymentMethod: str = Field(default="Electronic check", description="Billing payment method")
    InternetService: str = Field(default="Fiber optic", description="'DSL', 'Fiber optic', or 'No'")
    OnlineSecurity: str = Field(default="No", description="'Yes', 'No', or 'No internet service'")
    TechSupport: str = Field(default="No", description="'Yes', 'No', or 'No internet service'")
    OnlineBackup: str = Field(default="No", description="'Yes', 'No', or 'No internet service'")
    DeviceProtection: str = Field(default="No", description="'Yes', 'No', or 'No internet service'")
    StreamingTV: str = Field(default="Yes", description="'Yes', 'No', or 'No internet service'")
    StreamingMovies: str = Field(default="Yes", description="'Yes', 'No', or 'No internet service'")
    MultipleLines: str = Field(default="No", description="'Yes', 'No', or 'No phone service'")
    MonthlyCharges: float = Field(default=85.50, ge=10.0, le=200.0, description="Monthly subscription amount")
    TotalCharges: Optional[float] = Field(default=None, description="Cumulative charges (auto-calculated if omitted)")
    SupportTickets: int = Field(default=2, ge=0, description="Number of customer support tickets raised")
    PaymentFailures: int = Field(default=1, ge=0, description="Number of payment disputes or failures in last year")
    SatisfactionScore: float = Field(default=3.0, ge=1.0, le=5.0, description="Customer satisfaction rating 1 to 5")


class RiskDriver(BaseModel):
    feature: str
    impact: float
    direction: str


class PredictionResponse(BaseModel):
    customerID: str
    churn_probability: float
    churn_prediction: bool
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    retention_recommendation: str
    inference_time_ms: float


class ExplanationResponse(BaseModel):
    customerID: str
    churn_probability: float
    risk_level: str
    base_value: float
    top_risk_drivers: List[RiskDriver]
    top_retention_factors: List[RiskDriver]
    inference_time_ms: float


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerInput]


class BatchPredictionResponse(BaseModel):
    total_customers: int
    predicted_churners: int
    churn_rate: float
    results: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    algorithm: str
    version: str
