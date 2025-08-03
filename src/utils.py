import os
import sys
import dill
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_classification_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            logging_info = f"Training model: {model_name}"
            print(logging_info)

            params = param.get(model_name, {})

            gs = GridSearchCV(model, params, cv=3, scoring="accuracy", verbose=0)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_pred = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            report[model_name] = accuracy

        return report

    except Exception as e:
        raise CustomException(e, sys)
