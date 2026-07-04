from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "calibrated_penalty_model.joblib")

# 1. Load the ML Model globally when the server starts
print("Loading ML model...")
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
else:
    model = None
    print("WARNING: Model not found!")

# 2. Initialize FastAPI
app = FastAPI(
    title="Football Decision Intelligence API",
    description="API for predicting penalty outcomes",
    version="1.0.0"
)

# 3. Define the Request Schema (What data the user must send)
class PenaltyInput(BaseModel):
    shot_distance: float
    shot_angle: float
    is_shootout: int

# 4. Define the Endpoints
@app.get("/")
def home():
    """Health check endpoint"""
    return {"status": "online", "message": "Football Decision Intelligence API is running"}

@app.post("/predict/penalty")
def predict_penalty(data: PenaltyInput):
    """Predicts the probability of a penalty being scored"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
        
    # Convert input to DataFrame (models expect this format)
    input_df = pd.DataFrame([data.dict()])
    
    # Get prediction (probability of class 1: Goal)
    probability = model.predict_proba(input_df)[0, 1]
    
    return {
        "input": data.dict(),
        "probability_of_goal": round(float(probability), 4),
        "probability_as_percentage": f"{round(float(probability) * 100, 1)}%"
    }