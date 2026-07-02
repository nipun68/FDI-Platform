import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "features", "penalties_features.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

def main():
    print("Loading features...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Select Features
    features = ['shot_distance', 'shot_angle', 'is_shootout']
    target = 'is_goal'
    
    X = df[features]
    y = df[target]
    
    # 2. Train/Test Split (80% train, 20% test)
    
    # Because our sample is so small right now, we use a random split just to prove the pipeline works.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # 3. Model Training
    
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # 4. Predictions (Probabilities)
    # We want the probability of class 1 (Goal)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 5. Evaluation Metrics
   # Brier Score: 0 is perfect, 1 is worst. (Lower is better)
    # ROC-AUC: 0.5 is random guessing, 1.0 is perfect. (Higher is better)
    brier = brier_score_loss(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n--- Model Evaluation ---")
    print(f"Brier Score (Lower is better): {brier:.4f}")
    print(f"ROC-AUC (Higher is better): {roc_auc:.4f}")
    
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "baseline_penalty_model.joblib")
    joblib.dump(model, model_path)
    
    print(f"\n[SUCCESS] Model saved to: {model_path}")
    
    
    sample_data = pd.DataFrame({
        'shot_distance': [12.0],
        'shot_angle': [18.0],
        'is_shootout': [1]
    })
    
    prob_goal = model.predict_proba(sample_data)[0, 1]
    print(f"\n--- Sample Prediction ---")
    print(f"Given a standard shootout penalty, model predicts a {prob_goal * 100:.1f}% chance of a goal.")

if __name__ == "__main__":
    main()