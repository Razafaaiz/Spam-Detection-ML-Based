import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train():
    try:
        # Load dataset
        df = pd.read_csv("notebook/data/mail_data.csv")
        df = df.dropna()  # Remove missing values

        # Encode target variable
        df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})
        
        X = df['Message']
        y = df['Category']

        # Split into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Save train and test splits
        train_df = pd.DataFrame({'Message': X_train, 'Category': y_train})
        test_df = pd.DataFrame({'Message': X_test, 'Category': y_test})

        os.makedirs("artifacts", exist_ok=True)
        train_df.to_csv("artifacts/train_data.csv", index=False)
        test_df.to_csv("artifacts/test_data.csv", index=False)

        print("Train and test CSVs saved to artifacts/")

        # Vectorize text data
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Train model
        model = LogisticRegression()
        model.fit(X_train_vec, y_train)

        # Evaluate
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {acc:.2f}")

        # Save model and vectorizer
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open("artifacts/vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

        print("Training completed and files saved.")

    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    train()
