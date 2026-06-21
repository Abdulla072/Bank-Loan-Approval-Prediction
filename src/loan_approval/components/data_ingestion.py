import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from loan_approval.utils.logger import logger
from loan_approval.exception import CustomException


class DataIngestion:

    def __init__(self):

        self.raw_data_path = "data/raw/loan_approval_dataset.csv"

        self.train_data_path = "data/processed/train.csv"

        self.test_data_path = "data/processed/test.csv"

    def initiate_data_ingestion(self):

        try:

            logger.info("Data ingestion started.")

            df = pd.read_csv(self.raw_data_path)

            # Remove leading and trailing spaces from column names
            df.columns = df.columns.str.strip()

            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            train_df.to_csv(self.train_data_path, index=False)

            test_df.to_csv(self.test_data_path, index=False)

            logger.info(f"Train data saved at: {self.train_data_path}")

            logger.info(f"Test data saved at: {self.test_data_path}")

            logger.info("Data ingestion completed.")

            return (self.train_data_path, self.test_data_path)

        except Exception as e:

            logger.error(str(e))

            raise CustomException(e, sys)
