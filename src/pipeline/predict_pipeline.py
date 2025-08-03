import pickle
import os

class SpamClassifier:
    def __init__(self):
        model_path = "artifacts/model.pkl"
        vectorizer_path = "artifacts/vectorizer.pkl"

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError("Trained model or vectorizer not found.")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

    def predict(self, message):
        transformed = self.vectorizer.transform([message])
        prediction = self.model.predict(transformed)[0]
        return "Spam" if prediction == 1 else "Not Spam"


