from fastapi import FastAPI
from pydantic import BaseModel

from loan_approval.pipelines.prediction_pipeline import (
    PredictionPipeline,
)

app = FastAPI(
    title="Bank Loan Approval API",
    version="1.0.0",
)


class LoanData(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: float
    loan_amount: float
    loan_term: int
    cibil_score: int
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float


@app.get("/")
def home():

    return {"message": "Bank Loan Approval Prediction API"}


@app.post("/predict")
def predict(data: LoanData):

    pipeline = PredictionPipeline()

    result = pipeline.predict(data.model_dump())

    return {"loan_status": result}
