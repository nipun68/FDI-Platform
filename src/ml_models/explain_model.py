import pandas as pd
import os
import joblib
import shap
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic", "synthetic_penalties_10k.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "calibrated_penalty_model.joblib")

def main():
    print("Loading model and data...")
    df = pd.read_csv(DATA_PATH)
    
    calibrated_model = joblib.load(MODEL_PATH)
    base_model = calibrated_model.calibrated_classifiers_[0].estimator

    features = [col for col in df.columns if col != 'is_goal']
    X = df[features]

    X_encoded = pd.get_dummies(X)

    sample_index = 0
    sample_penalty = X_encoded.iloc[[sample_index]]
    actual_outcome = df.iloc[sample_index]['is_goal']
    
    print(f"\nExplaining penalty at index {sample_index}")
    print(f"Actual Outcome: {'Goal' if actual_outcome == 1 else 'No Goal'}")
    
    print("\nCalculating SHAP values (this may take a few seconds)...")
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(sample_penalty)
    
    print("\n--- SHAP Explanation ---")
    print("Base Value (Average prediction for all penalties):", explainer.expected_value)

    feature_contributions = list(zip(sample_penalty.columns, shap_values[0]))
    feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for feature, contribution in feature_contributions[:5]:
        direction = "increased" if contribution > 0 else "decreased"
        val = sample_penalty[feature].values[0]
        print(f"Feature '{feature}' (Value: {val}) {direction} the probability by {abs(contribution):.4f}")

    plt.figure()
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value, 
        shap_values[0], 
        sample_penalty.iloc[0],
        feature_names=X_encoded.columns,
        show=False,
        max_display=10
    )

    plots_dir = os.path.join(PROJECT_ROOT, "data", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "shap_explanation.png")
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"\n✅ SHAP waterfall plot saved to: {plot_path}")

if __name__ == "__main__":
    main()