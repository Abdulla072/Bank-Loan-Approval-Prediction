from loan_approval.pipelines.prediction_pipeline import PredictionPipeline

sample_data = {
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 5000000,
    "loan_amount": 1000000,
    "loan_term": 12,
    "cibil_score": 750,
    "residential_assets_value": 2000000,
    "commercial_assets_value": 0,
    "luxury_assets_value": 500000,
    "bank_asset_value": 1000000,
}

pipeline = PredictionPipeline()

result = pipeline.predict(sample_data)

print(f"Loan Status: {result}")
