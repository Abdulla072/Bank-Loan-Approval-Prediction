import pandas as pd

from loan_approval.utils.common import load_object


class PredictionPipeline:

    def __init__(self):

        self.model = load_object("artifacts/model.pkl")
        self.preprocessor = load_object("artifacts/preprocessor.pkl")
        self.label_encoder = load_object("artifacts/label_encoder.pkl")

    def predict(self, data: dict):

        df = pd.DataFrame([data])

        transformed_data = self.preprocessor.transform(df)

        prediction = self.model.predict(transformed_data)

        result = self.label_encoder.inverse_transform(prediction)

        return result[0]
