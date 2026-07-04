from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "calibrated_penalty_model.joblib")

print("Loading ML model...")
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    trained_features = model.calibrated_classifiers_[0].estimator.feature_names_
    print("Model loaded successfully!")
else:
    model = None
    trained_features = []
    print("WARNING: Model not found!")

app = FastAPI(
    title="Football Decision Intelligence API",
    description="Enterprise API for predicting penalty outcomes using advanced 7-category features.",
    version="2.0.0"
)

class PenaltyInput(BaseModel):
    player_penalty_conversion_rate: float
    career_penalty_attempts: int
    recent_penalty_form: float
    is_shootout: int
    match_stage: str
    minute_of_match: int
    team_pressure_state: str
    run_up_style: str
    pause_before_shot: int
    shot_telegraph_level: int
    goalkeeper_penalty_save_rate: float
    goalkeeper_diving_bias: str
    keeper_experience_level: int
    shot_direction: str
    shot_height: str
    corner_precision: float
    goalkeeper_dive_direction: str
    gk_correct_guess: int
    reaction_time_ms: float
    shooter_vs_keeper_history: int
    psychological_advantage_index: int

@app.get("/")
def home():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict/penalty")
def predict_penalty(data: PenaltyInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
        
    input_df = pd.DataFrame([data.dict()])
    input_encoded = pd.get_dummies(input_df)
    input_aligned = input_encoded.reindex(columns=trained_features, fill_value=0)
    
    probability = model.predict_proba(input_aligned)[0, 1]
    
    return {
        "probability_of_goal": round(float(probability), 4),
        "probability_as_percentage": f"{round(float(probability) * 100, 1)}%",
        "key_context": {
            "match_stage": data.match_stage,
            "is_shootout": bool(data.is_shootout),
            "gk_correct_guess": bool(data.gk_correct_guess)
        }
    }