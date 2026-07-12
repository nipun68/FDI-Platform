import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_ingestion.generate_synthetic_data import main as gen_data
from src.ml_models.train_striker_model import main as train_striker
from src.ml_models.train_goalkeeper_model import main as train_gk
from src.ml_models.train_outcome_model import main as train_outcome

def test_data_pipeline():
    """Tests if synthetic data can be generated without errors."""
    try:
        gen_data()
        assert os.path.exists("data/synthetic/synthetic_penalties_50k.csv")
    except Exception as e:
        assert False, f"Data generation failed: {e}"

def test_model_training():
    """Tests if the 3 interaction engine models can be trained."""
    try:
        train_striker()
        train_gk()
        train_outcome()
        assert os.path.exists("models/striker_agent_model.joblib")
        assert os.path.exists("models/gk_agent_model.joblib")
        assert os.path.exists("models/outcome_interaction_model.joblib")
    except Exception as e:
        assert False, f"Model training failed: {e}"

if __name__ == "__main__":
    test_data_pipeline()
    test_model_training()
    print("All tests passed!")