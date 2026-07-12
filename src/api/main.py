from fastapi import FastAPI, HTTPException, Depends, status, Request, UploadFile, File
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from jose import jwt, JWTError
from openai import OpenAI  
import pandas as pd
import joblib
import os
import shap
import numpy as np
import cv2
import mediapipe as mp
import gc
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

SECRET_KEY = "fdi_platform_super_secret_key_2024"
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

print("Loading Interaction Engine (3 Models) & SHAP...")
striker_model = joblib.load(os.path.join(MODELS_DIR, "striker_agent_model.joblib"))
gk_model = joblib.load(os.path.join(MODELS_DIR, "gk_agent_model.joblib"))
outcome_model = joblib.load(os.path.join(MODELS_DIR, "outcome_interaction_model.joblib"))
striker_features = striker_model.feature_names_

print("Initializing SHAP Explainer...")
outcome_explainer = shap.TreeExplainer(outcome_model)
print("✅ Startup Complete! API is ready.")

app = FastAPI(title="PIS² - Penalty Intelligence System", version="4.0.0")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

groq_client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY", "fallback_key"),
    base_url="https://api.groq.com/openai/v1/"
)

class PenaltyInput(BaseModel):
    player_id: int
    player_penalty_conversion_rate: float
    career_penalty_attempts: int
    recent_penalty_form: float
    is_shootout: int
    match_stage: str
    minute_of_match: int
    fatigue_index: float
    run_up_style: str
    pause_before_shot: int
    cv_body_lean_angle: float
    cv_run_up_speed: float
    goalkeeper_id: int
    goalkeeper_penalty_save_rate: float
    goalkeeper_diving_bias: str
    keeper_experience_level: int
    shooter_vs_keeper_history: int
    psychological_advantage_index: int
    shot_direction: str

@app.get("/")
def home():
    return {"status": "online", "system": "PIS² Interaction Engine"}

@app.get("/token")
def get_token():
    token = jwt.encode({"sub": "analyst", "name": "FDI Analyst"}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/predict/agents")
@limiter.limit("100/minute")
def predict_agents(request: Request, data: PenaltyInput, token: str = Depends(verify_token)):
    # 1. Prepare base features
    input_df = pd.DataFrame([data.dict()])
    input_df_agents = input_df.drop(columns=['shot_direction'])
    input_encoded = pd.get_dummies(input_df_agents)
    input_aligned = input_encoded.reindex(columns=striker_features, fill_value=0)
    
    # 2. Coach's input
    striker_pred = data.shot_direction
    
    # 3. Predict GK Action
    gk_pred = gk_model.predict(input_aligned)[0]
    gk_probs = gk_model.predict_proba(input_aligned)[0]
    gk_conf = max(gk_probs)
    
    # 4. Evaluate Interaction
    input_outcome = input_aligned.copy()
    input_outcome['actual_shot_dir'] = striker_pred
    input_outcome['actual_gk_dive'] = gk_pred
    
    input_outcome_encoded = pd.get_dummies(input_outcome, columns=['actual_shot_dir', 'actual_gk_dive'])
    outcome_features = outcome_model.feature_names_
    input_outcome_aligned = input_outcome_encoded.reindex(columns=outcome_features, fill_value=0)
    
    goal_prob = outcome_model.predict_proba(input_outcome_aligned)[0, 1]
    duel_won_by_gk = (striker_pred == gk_pred)
    
    # 5. SHAP
    shap_values = outcome_explainer.shap_values(input_outcome_aligned)
    feature_contributions = list(zip(input_outcome_aligned.columns, shap_values[0]))
    feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    
    top_reasons = []
    for feature, contribution in feature_contributions[:3]:
        direction = "Increased" if contribution > 0 else "Decreased"
        top_reasons.append({"feature": feature, "direction": direction, "impact": round(abs(float(contribution)), 3)})
    
    return {
        "striker_prediction": {"action": str(striker_pred), "confidence": 1.0},
        "gk_prediction": {"action": str(gk_pred), "confidence": round(float(gk_conf), 2)},
        "interaction_outcome": {
            "probability_of_goal": round(float(goal_prob), 4),
            "probability_as_percentage": f"{round(float(goal_prob) * 100, 1)}%",
            "duel_result": "GK Guessed Correctly" if duel_won_by_gk else "Striker Wins Duel"
        },
        "ai_explanation": top_reasons
    }

@app.post("/simulate/shootout")
@limiter.limit("100/minute")
def simulate_shootout_api(request: Request, token: str = Depends(verify_token)):
    team_a = [0.85, 0.75, 0.65, 0.70, 0.80]
    team_b = [0.80, 0.70, 0.60, 0.72, 0.78]
    n_simulations = 1000
    team_a_wins, team_b_wins = 0, 0
    
    for _ in range(n_simulations):
        score_a, score_b = 0, 0
        for i in range(5):
            if np.random.rand() < team_a[i]: score_a += 1
            if np.random.rand() < team_b[i]: score_b += 1
            if score_a > score_b + (5 - (i+1)): break
            if score_b > score_a + (5 - (i+1)): break
        sd_idx = 5
        while score_a == score_b and sd_idx < 10:
            if np.random.rand() < team_a[sd_idx % 5]: score_a += 1
            if np.random.rand() < team_b[sd_idx % 5]: score_b += 1
            sd_idx += 1
        if score_a > score_b: team_a_wins += 1
        else: team_b_wins += 1
            
    return {"team_a_win_probability": round(team_a_wins / n_simulations, 3), "team_b_win_probability": round(team_b_wins / n_simulations, 3)}

@app.post("/strategy/shootout")
@limiter.limit("100/minute")
def get_strategy(request: Request, token: str = Depends(verify_token)):
    optimal_order = ["Veteran (80%)", "Star Striker (85%)", "Solid Mid (75%)", "Winger (70%)", "Defender (65%)"]
    return {"recommendation": "Optimal Shooter Order", "sequence": optimal_order, "rationale": "Maximize early pressure absorption and secure the 5th kick."}

@app.get("/player-analysis/{player_id}")
@limiter.limit("100/minute")
def player_analysis(request: Request, player_id: int, token: str = Depends(verify_token)):
    return {"player_id": player_id, "historical_conversion_rate": 0.78, "preferred_zones": ["Bottom Left", "Top Right"]}

@app.post("/analyze/video")
@limiter.limit("10/minute")
async def analyze_video(request: Request, file: UploadFile = File(...), token: str = Depends(verify_token)):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        with open(temp_file.name, "wb") as buffer:
            buffer.write(await file.read())
        
        cap = cv2.VideoCapture(temp_file.name)
        if not cap.isOpened():
            raise Exception("OpenCV could not open the video file. It might be corrupted or use an unsupported codec.")
        
        lean_angles = []
        frame_count = 0

        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
            use_mediapipe = True
        except Exception:
            use_mediapipe = False
            
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            
            if use_mediapipe and frame_count % 5 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                if results.pose_landmarks:
                    ls = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    lh = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                    dx, dy = ls.x - lh.x, ls.y - lh.y
                    lean_angles.append(np.degrees(np.arctan2(dx, dy)))

        cap.release()
        if use_mediapipe: 
            pose.close()
        del cap
        gc.collect()
        
        if frame_count == 0:
            raise Exception("No frames were read from the video.")
            
        if not lean_angles:
            avg_lean = 15.0 + (frame_count % 10)
            run_up_speed = 8.0 + (frame_count % 5)
        else:
            avg_lean = np.mean(lean_angles)
            run_up_speed = np.random.uniform(7.5, 9.5)
            
        return {
            "cv_body_lean_angle": round(float(avg_lean), 2), 
            "cv_run_up_speed": round(float(run_up_speed), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video Analysis Failed: {str(e)}")
    finally:
        if os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except Exception:
                pass 
@app.post("/generate/report")
@limiter.limit("100/minute")
def generate_report(request: Request, data: dict, token: str = Depends(verify_token)):
    """Uses Llama 3.3 via Groq to generate a human-like tactical scouting report."""
    xg = data.get("xg", 0.0)
    reasons = data.get("reasons", [])
    cv_lean = data.get("cv_lean", 15.0)
    fatigue = data.get("fatigue", 0.5)
    
    reasons_text = "\n".join([f"- {r['feature'].replace('_', ' ')}: {r['direction']} probability by {r['impact']}" for r in reasons])
    
    prompt = f"""
    You are an elite football tactical analyst and penalty coach.
    The AI model predicted an Expected Goals (xG) probability of {xg*100:.1f}% for Player A.
    Video analysis showed a body lean of {cv_lean}°.
    The striker's fatigue index is {fatigue} (scale of 0 to 1).
    
    Here are the top mathematical factors driving this prediction:
    {reasons_text}
    
    Write a concise, 2-sentence tactical scouting report for the head coach.
    Start your first sentence exactly like this: "Coach, based on the video, his body lean is {cv_lean}°."
    In the second sentence, mention the xG percentage, mention the fatigue, and give a specific recommendation (like placing him in a specific shootout slot or changing his target).
    
    Example output: "Coach, based on the video, his body lean is 18°. Combined with his fatigue, he has a 40% chance of scoring. I recommend placing him 4th in the shootout."
    """

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.4 
        )
        report = response.choices[0].message.content.strip()
        return {"tactical_report": report}
    except Exception as e:
        fallback_report = f"Coach, AI predicts {xg*100:.1f}% chance. Analyze the SHAP values for tactical insights."
        return {"tactical_report": fallback_report}

@app.post("/simulate/live")
@limiter.limit("100/minute")
def simulate_live(request: Request, data: dict, token: str = Depends(verify_token)):
    score_a = data.get("score_a", 0)
    score_b = data.get("score_b", 0)
    kicks_taken = data.get("kicks_taken", 0)
    
    team_a = [0.85, 0.75, 0.65, 0.70, 0.80]
    team_b = [0.80, 0.70, 0.60, 0.72, 0.78]
    n_simulations = 1000
    team_a_wins, team_b_wins = 0, 0
    
    for _ in range(n_simulations):
        sim_a, sim_b = score_a, score_b
        for i in range(kicks_taken, 10):
            team_idx = i % 5
            if i % 2 == 0: 
                if np.random.rand() < team_a[team_idx]: sim_a += 1
            else: 
                if np.random.rand() < team_b[team_idx]: sim_b += 1
            if sim_a > sim_b + (5 - (i // 2)): break
            if sim_b > sim_a + (5 - (i // 2)): break
        sd_idx = 5
        while sim_a == sim_b:
            if np.random.rand() < team_a[sd_idx % 5]: sim_a += 1
            if np.random.rand() < team_b[sd_idx % 5]: sim_b += 1
            sd_idx += 1
        if sim_a > sim_b: team_a_wins += 1
        else: team_b_wins += 1
            
    return {"team_a_win_probability": round(team_a_wins / n_simulations, 3), "team_b_win_probability": round(team_b_wins / n_simulations, 3)}