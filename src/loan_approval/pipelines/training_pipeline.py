from loan_approval.components.data_ingestion import DataIngestion
from loan_approval.components.data_validation import DataValidation
from loan_approval.components.data_transformation import DataTransformation
from loan_approval.components.model_trainer import ModelTrainer


if __name__ == "__main__":

    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()

    validation = DataValidation()
    validation.validate_data()

    transformation = DataTransformation()

    X_train, X_test, y_train, y_test = (
        transformation.initiate_data_transformation()
    )

    trainer = ModelTrainer()

    trainer.initiate_model_training(
        X_train,
        X_test,
        y_train,
        y_test
    )