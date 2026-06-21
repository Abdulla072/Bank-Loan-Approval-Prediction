import sys
import pandas as pd

from loan_approval.utils.logger import logger
from loan_approval.utils.common import read_yaml
from loan_approval.exception import CustomException


class DataValidation:

    def __init__(self):
        self.train_data_path = "data/processed/train.csv"
        self.schema_path = "config/schema.yaml"

    def validate_data(self):

        try:

            df = pd.read_csv(self.train_data_path)

            schema = read_yaml(self.schema_path)

            expected_columns = schema["columns"]

            # Column validation
            missing_columns = set(expected_columns.keys()) - set(df.columns)

            if missing_columns:
                raise ValueError(
                    f"Missing columns: {missing_columns}"
                )

            # Data type validation
            for column, expected_dtype in expected_columns.items():

                actual_dtype = str(df[column].dtype)

                if expected_dtype == "object":

                    if actual_dtype not in ["object", "string", "str"]:
                        raise ValueError(
                            f"{column}: expected text type, got {actual_dtype}"
                        )

                else:

                    if actual_dtype != expected_dtype:
                        raise ValueError(
                            f"{column}: expected {expected_dtype}, got {actual_dtype}"
                        )

            # Missing value validation
            total_missing = df.isnull().sum().sum()

            if total_missing > 0:
                raise ValueError(
                    f"Missing values found: {total_missing}"
                )

            # Duplicate validation
            duplicate_rows = df.duplicated().sum()

            if duplicate_rows > 0:
                raise ValueError(
                    f"Duplicate rows found: {duplicate_rows}"
                )

            logger.info("Data validation completed.")

            return True

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)