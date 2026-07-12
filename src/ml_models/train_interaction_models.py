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
    print("Loading data for Interaction Engine...")
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
    
    cat_features = ['match_stage', 'run_up_style', 'goalkeeper_diving_bias']
    
    X_base = df[base_features]
    X_encoded = pd.get_dummies(X_base, columns=cat_features)
    
    print("Training Striker Agent (Shot Direction)...")
    y_shot = df['shot_direction']
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_encoded, y_shot, test_size=0.2, random_state=42)
    striker_model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=0, random_seed=42)
    striker_model.fit(X_train_s, y_train_s)
    print(f"Striker Accuracy: {accuracy_score(y_test_s, striker_model.predict(X_test_s)):.4f}")
    joblib.dump(striker_model, os.path.join(MODELS_DIR, "striker_agent_model.joblib"))

    print("Training Goalkeeper Agent (Dive Direction)...")
    y_gk = df['goalkeeper_dive_direction']
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_encoded, y_gk, test_size=0.2, random_state=42)
    gk_model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=0, random_seed=42)
    gk_model.fit(X_train_g, y_train_g)
    print(f"GK Accuracy: {accuracy_score(y_test_g, gk_model.predict(X_test_g)):.4f}")
    joblib.dump(gk_model, os.path.join(MODELS_DIR, "gk_agent_model.joblib"))

    print("Training Outcome Model (Interaction)...")
    X_outcome = X_encoded.copy()
    X_outcome['predicted_shot_dir'] = striker_model.predict(X_encoded)
    X_outcome['predicted_gk_dive'] = gk_model.predict(X_encoded)
    
    y_goal = df['is_goal']
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_outcome, y_goal, test_size=0.2, random_state=42)
    outcome_model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=0, random_seed=42)
    outcome_model.fit(X_train_o, y_train_o)
    joblib.dump(outcome_model, os.path.join(MODELS_DIR, "outcome_interaction_model.joblib"))
    print("\n✅ Interaction Engine (3 Models) saved successfully.")

if __name__ == "__main__":
    main()