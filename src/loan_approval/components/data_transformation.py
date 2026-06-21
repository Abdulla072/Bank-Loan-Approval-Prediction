import os
import pandas as pd
from loan_approval.utils.logger import logger

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from loan_approval.utils.common import save_object


class DataTransformation:

    def __init__(self):

        self.train_data_path = "data/processed/train.csv"
        self.test_data_path = "data/processed/test.csv"

        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def get_data_transformer_object(self):

        categorical_cols = [
            "education",
            "self_employed"
        ]

        numerical_cols = [
            "no_of_dependents",
            "income_annum",
            "loan_amount",
            "loan_term",
            "cibil_score",
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value"
        ]

        numerical_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler())
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, numerical_cols),
                ("cat", categorical_pipeline, categorical_cols)
            ]
        )

        return preprocessor

    def initiate_data_transformation(self):

        train_df = pd.read_csv(self.train_data_path)
        test_df = pd.read_csv(self.test_data_path)

        train_df = train_df.drop(columns=["loan_id"])
        test_df = test_df.drop(columns=["loan_id"])

        X_train = train_df.drop(columns=["loan_status"])
        y_train = train_df["loan_status"]

        X_test = test_df.drop(columns=["loan_status"])
        y_test = test_df["loan_status"]

        preprocessor = self.get_data_transformer_object()

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        save_object(
            self.preprocessor_path,
            preprocessor
        )

        logger.info("Data transformation completed.")

        return (
            X_train_transformed,
            X_test_transformed,
            y_train,
            y_test
        )