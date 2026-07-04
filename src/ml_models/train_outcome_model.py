import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic", "synthetic_penalties_10k.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

def main():
    print("Loading ULTRA-REALISTIC features (10k)...")
    df = pd.read_csv(DATA_PATH)

    features = [col for col in df.columns if col != 'is_goal']
    target = 'is_goal'
    
    X = df[features]
    y = df[target]

    X_encoded = pd.get_dummies(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    print("\nTraining Logistic Regression model on Advanced Features...")
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    brier = brier_score_loss(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n--- Model Evaluation ---")
    print(f"Brier Score (Lower is better): {brier:.4f}")
    print(f"ROC-AUC (Higher is better): {roc_auc:.4f}")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "baseline_penalty_model.joblib")
    joblib.dump(model, model_path)
    print(f"\n[SUCCESS] Baseline model saved to: {model_path}")

if __name__ == "__main__":
    main()