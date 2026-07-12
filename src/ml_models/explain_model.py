import pandas as pd
import os
import joblib
import shap
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic", "synthetic_penalties_50k.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

def main():
    print("Loading Outcome Interaction Model & Data...")
    df = pd.read_csv(DATA_PATH)
    
    # Load all 3 models
    striker_model = joblib.load(os.path.join(MODELS_DIR, "striker_agent_model.joblib"))
    gk_model = joblib.load(os.path.join(MODELS_DIR, "gk_agent_model.joblib"))
    outcome_model = joblib.load(os.path.join(MODELS_DIR, "outcome_interaction_model.joblib"))
    
    base_features = [
        'player_id', 'player_penalty_conversion_rate', 'career_penalty_attempts', 
        'recent_penalty_form', 'is_shootout', 'match_stage', 'minute_of_match', 
        'fatigue_index', 'run_up_style', 'pause_before_shot', 
        'cv_body_lean_angle', 'cv_run_up_speed', 'goalkeeper_id', 
        'goalkeeper_penalty_save_rate', 'goalkeeper_diving_bias', 
        'keeper_experience_level', 'shooter_vs_keeper_history', 
        'psychological_advantage_index'
    ]
    
    X = df[base_features].head(100) # Explain top 100 to save memory
    X_encoded = pd.get_dummies(X, columns=['match_stage', 'run_up_style', 'goalkeeper_diving_bias'])
    
    # Generate Agent predictions for the Outcome model
    X_encoded['predicted_shot_dir'] = striker_model.predict(X_encoded)
    X_encoded['predicted_gk_dive'] = gk_model.predict(X_encoded)
    
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(outcome_model)
    shap_values = explainer.shap_values(X_encoded)
    
    # Plot summary for the top features
    plt.figure()
    shap.summary_plot(shap_values, X_encoded, show=False)
    
    plots_dir = os.path.join(PROJECT_ROOT, "data", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "shap_outcome_summary.png")
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"\n✅ SHAP summary plot saved to: {plot_path}")

if __name__ == "__main__":
    main()