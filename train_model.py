import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocess_data import load_and_preprocess
import numpy as np

def train_and_eval():
    # 1. Load Processed Data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # 2. Transform Target Variable (Regression -> Classification)
    # We'll define "High Performance" as FinalGrade >= 80 (Data Median)
    # 1: High Performance, 0: Average/Low Performance
    threshold = 80
    print(f"\nTransforming target variable: Binary Classification (Threshold >= {threshold})")
    
    y_train_class = (y_train >= threshold).astype(int)
    y_test_class = (y_test >= threshold).astype(int)

    print(f"Class distribution in Train set: {np.bincount(y_train_class)}")
    
    # 3. Initialize Model
    # Using Logistic Regression as requested
    print("\nInitializing Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)

    # 4. Train Model
    print("Training model...")
    model.fit(X_train, y_train_class)

    # 5. Predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)

    # 6. Evaluation
    acc = accuracy_score(y_test_class, y_pred)
    print(f"\nModel Accuracy: {acc:.2f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test_class, y_pred, target_names=['Average/Low', 'High']))

    # 7. Save Model
    model_filename = "student_performance_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    train_and_eval()
