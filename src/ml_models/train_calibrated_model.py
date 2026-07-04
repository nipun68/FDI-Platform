import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier
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
    
  
    X_encoded = pd.get_dummies(X, columns=[
        'match_stage', 'team_pressure_state', 'run_up_style', 
        'goalkeeper_diving_bias', 'shot_direction', 'shot_height', 
        'goalkeeper_dive_direction'
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    print("\nTraining CatBoost on Advanced 7-Category Features...")
    
    base_model = CatBoostClassifier(
        iterations=500, 
        depth=6,        
        learning_rate=0.05, 
        verbose=0, 
        random_seed=42
    )
    base_model.fit(X_train, y_train)
    
    print("Calibrating probabilities...")
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n--- Ultimate Model Evaluation ---")
    print(f"Brier Score (Lower is better): {brier:.4f}")
    print(f"ROC-AUC (Higher is better): {roc_auc:.4f}")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "calibrated_penalty_model.joblib")
    joblib.dump(calibrated_model, model_path)
    
    
    # We must pass the data through get_dummies first to match training format
    sample_data = pd.DataFrame({
        'player_penalty_conversion_rate': [0.85],
        'career_penalty_attempts': [30],
        'recent_penalty_form': [0.8],
        'is_shootout': [1],
        'match_stage': ['Final'],
        'minute_of_match': [120],
        'team_pressure_state': ['Drawing'],
        'run_up_style': ['Stutter'],
        'pause_before_shot': [1],
        'shot_telegraph_level': [1],
        'goalkeeper_penalty_save_rate': [0.25],
        'goalkeeper_diving_bias': ['Left'],
        'keeper_experience_level': [5],
        'shot_direction': ['Left'],
        'shot_height': ['Low'],
        'corner_precision': [0.95],
        'goalkeeper_dive_direction': ['Right'], # GK dived wrong way!
        'gk_correct_guess': [0],
        'reaction_time_ms': [220],
        'shooter_vs_keeper_history': [0],
        'psychological_advantage_index': [90]
    })
    
    sample_encoded = pd.get_dummies(sample_data)

    sample_aligned = sample_encoded.reindex(columns=X_train.columns, fill_value=0)
    
    prob_goal = calibrated_model.predict_proba(sample_aligned)[0, 1]
    print(f"\n--- Sample Prediction ---")
    print(f"Elite shooter, Final, GK dives wrong way -> {prob_goal * 100:.1f}% chance of goal.")

if __name__ == "__main__":
    main()