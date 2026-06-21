import sys

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from loan_approval.utils.logger import logger
from loan_approval.utils.common import save_object
from loan_approval.exception import CustomException


class ModelTrainer:

    def __init__(self):

        self.model_path = "artifacts/model.pkl"
        self.label_encoder_path = "artifacts/label_encoder.pkl"

    def initiate_model_training(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
    ):

        try:

            logger.info("Model training started.")

            label_encoder = LabelEncoder()

            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)

            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
                "Decision Tree": DecisionTreeClassifier(
                    random_state=42,
                ),
                "Random Forest": RandomForestClassifier(
                    random_state=42,
                ),
            }

            best_model = None
            best_model_name = ""
            best_accuracy = 0.0

            for model_name, model in models.items():

                logger.info(f"Training {model_name}")

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)

                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)

                logger.info(f"{model_name} accuracy: {accuracy:.4f}")

                logger.info(f"{model_name} precision: {precision:.4f}")

                logger.info(f"{model_name} recall: {recall:.4f}")

                logger.info(f"{model_name} f1-score: {f1:.4f}")

                logger.info(f"{model_name} confusion matrix:\n{cm}")

                if accuracy > best_accuracy:

                    best_accuracy = accuracy
                    best_model = model
                    best_model_name = model_name

            save_object(
                self.model_path,
                best_model,
            )

            save_object(
                self.label_encoder_path,
                label_encoder,
            )

            logger.info(f"Best model: {best_model_name}")

            logger.info(f"Best accuracy: {best_accuracy:.4f}")

            logger.info("Model training completed.")

            return best_accuracy

        except Exception as e:

            logger.error(str(e))

            raise CustomException(e, sys)
