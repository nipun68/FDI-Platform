import pandas as pd
import os
import joblib
import shap
import matplotlib.pyplot as plt
                               
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "features", "penalties_features.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "calibrated_penalty_model.joblib")

def main():
    print("Loading model and data...")
    df = pd.read_csv(DATA_PATH)
    
    calibrated_model = joblib.load(MODEL_PATH)
    
    base_model = calibrated_model.calibrated_classifiers_[0].estimator
    
    features = ['shot_distance', 'shot_angle', 'is_shootout']
    X = df[features]
    
    sample_index = 0
    sample_penalty = X.iloc[[sample_index]]
    actual_outcome = df.iloc[sample_index]['is_goal']
    player_name = df.iloc[sample_index]['player_name']
    
    print(f"\nExplaining penalty for: {player_name}")
    print(f"Actual Outcome: {'Goal' if actual_outcome == 1 else 'No Goal'}")
    print(f"Features: \n{sample_penalty}")
    

    print("\nCalculating SHAP values (this may take a few seconds)...")
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(sample_penalty)
    
    
    print("\n--- SHAP Explanation ---")
    print("Base Value (Average prediction for all penalties):", explainer.expected_value)
    
    
    for i, feature in enumerate(features):
        # shap_values[0][i] is the SHAP value for the i-th feature
        contribution = shap_values[0][i]
        direction = "increased" if contribution > 0 else "decreased"
        print(f"Feature '{feature}' (Value: {sample_penalty[feature].values[0]}) {direction} the probability by {abs(contribution):.4f}")
          
    plt.figure()
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value, 
        shap_values[0], 
        sample_penalty.iloc[0],
        feature_names=features,
        show=False
    )

    plots_dir = os.path.join(PROJECT_ROOT, "data", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "shap_explanation.png")
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"\n✅ SHAP waterfall plot saved to: {plot_path}")

if __name__ == "__main__":
    main()