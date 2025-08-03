import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_classification_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Logistic Regression": LogisticRegression(max_iter=200),
                "Naive Bayes": MultinomialNB(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                "Logistic Regression": {"C": [0.1, 1, 10]},
                "Naive Bayes": {},
                "Random Forest": {"n_estimators": [50, 100], "max_depth": [None, 10, 20]},
                "Decision Tree": {"max_depth": [None, 10, 20]},
                "Gradient Boosting": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
                "XGBoost": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]},
                "AdaBoost": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}
            }

            model_report: dict = evaluate_classification_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            # Identify the best model
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with accuracy >= 60%")

            logging.info(f"Best model selected: {best_model_name} with accuracy: {best_model_score}")

            # Fit the best model again on all training data (if required)
            best_model.fit(X_train, y_train)

            # ✅ Ensure artifacts directory exists
            artifacts_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # ✅ Save the best model
            from src.utils import save_object

            save_object(file_path="artifacts/preprocessor.pkl", obj=preprocessor) # type: ignore
            save_object(file_path="artifacts/model.pkl", obj=model) # type: ignore


            # Make predictions and log results
            predictions = best_model.predict(X_test)
            acc = accuracy_score(y_test, predictions)

            logging.info(f"Final Accuracy on test set: {acc}")
            logging.info(f"Classification Report:\n{classification_report(y_test, predictions)}")

            return acc

        except Exception as e:
            raise CustomException(e, sys)

