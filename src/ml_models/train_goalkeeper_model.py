import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic", "synthetic_penalties_50k.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

def main():
    print("Loading data for Goalkeeper Agent...")
    df = pd.read_csv(DATA_PATH)
    
    base_features = [
        'player_id', 'player_penalty_conversion_rate', 'career_penalty_attempts', 
        'recent_penalty_form', 'is_shootout', 'match_stage', 'minute_of_match', 
        'fatigue_index', 'run_up_style', 'pause_before_shot', 
        'cv_body_lean_angle', 'cv_run_up_speed', 'goalkeeper_id', 
        'goalkeeper_penalty_save_rate', 'goalkeeper_diving_bias', 
        'keeper_experience_level', 'shooter_vs_keeper_history', 
        'psychological_advantage_index'
    ]
    
    X = df[base_features]
    X_encoded = pd.get_dummies(X, columns=['match_stage', 'run_up_style', 'goalkeeper_diving_bias'])
    y = df['gk_dive_direction'] # Target: Where will the GK dive?
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    print("Training Goalkeeper Agent (CatBoost)...")
    model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=0, random_seed=42)
    model.fit(X_train, y_train)
    
    print(f"GK Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "gk_agent_model.joblib"))
    print("✅ Goalkeeper Model saved.")

if __name__ == "__main__":
    main()