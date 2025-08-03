import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Returns a TF-IDF vectorizer as the transformation object for text data.
        """
        try:
            tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=5000  # You can tune this
            )
            return tfidf_vectorizer
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Successfully loaded training and testing data for transformation")

            # Features and target
            input_feature_train = train_df['text']
            target_feature_train = train_df['label']

            input_feature_test = test_df['text']
            target_feature_test = test_df['label']

            # Encode labels (ham/spam → 0/1)
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(target_feature_train)
            y_test = label_encoder.transform(target_feature_test)

            # Get and apply transformer (TF-IDF)
            transformer = self.get_data_transformer_object()
            X_train = transformer.fit_transform(input_feature_train)
            X_test = transformer.transform(input_feature_test)

            logging.info("TF-IDF transformation applied to text data")

            # Save transformer
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=transformer
            )
            logging.info("TF-IDF vectorizer saved to disk")

            # Return arrays
            return (
                np.c_[X_train.toarray(), y_train],
                np.c_[X_test.toarray(), y_test],
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
