import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic", "synthetic_penalties_5k.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

def main():
    print("Loading SCALED synthetic features (5,000 penalties)...")
    df = pd.read_csv(DATA_PATH)

    features = ['shot_distance', 'shot_angle', 'is_shootout']
    target = 'is_goal'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")

    print("\nTraining CatBoost model on scaled data...")
    base_model = CatBoostClassifier(
        iterations=200, 
        depth=4,        
        learning_rate=0.05, 
        verbose=0, 
        random_seed=42
    )
    base_model.fit(X_train, y_train)

    print("Calibrating probabilities using Isotonic Regression...")
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5) 
    calibrated_model.fit(X_train, y_train)

    y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n--- Scaled Model Evaluation ---")
    print(f"Brier Score (Lower is better): {brier:.4f}")
    print(f"ROC-AUC (Higher is better): {roc_auc:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "calibrated_penalty_model.joblib")
    joblib.dump(calibrated_model, model_path)
    print(f"\n✅ Scaled model saved to: {model_path}")

    sample_data = pd.DataFrame({
        'shot_distance': [11.0],
        'shot_angle': [22.0],
        'is_shootout': [1]
    })
    
    prob_goal = calibrated_model.predict_proba(sample_data)[0, 1]
    print(f"\n--- Sample Prediction ---")
    print(f"Given a standard shootout penalty, model now predicts a {prob_goal * 100:.1f}% chance of a goal.")

if __name__ == "__main__":
    main()