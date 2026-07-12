import pandas as pd
import os
import joblib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

def simulate_shootout():
    print("Loading Interaction Engine for simulation...")
    striker_model = joblib.load(os.path.join(MODELS_DIR, "striker_agent_model.joblib"))
    gk_model = joblib.load(os.path.join(MODELS_DIR, "gk_agent_model.joblib"))
    outcome_model = joblib.load(os.path.join(MODELS_DIR, "outcome_interaction_model.joblib"))
    team_a = [
        {'name': 'Star Striker', 'player_penalty_conversion_rate': 0.85, 'career_penalty_attempts': 30, 'recent_penalty_form': 0.8},
        {'name': 'Solid Mid', 'player_penalty_conversion_rate': 0.75, 'career_penalty_attempts': 15, 'recent_penalty_form': 0.7},
        {'name': 'Defender', 'player_penalty_conversion_rate': 0.65, 'career_penalty_attempts': 5, 'recent_penalty_form': 0.6},
        {'name': 'Winger', 'player_penalty_conversion_rate': 0.70, 'career_penalty_attempts': 10, 'recent_penalty_form': 0.65},
        {'name': 'Veteran', 'player_penalty_conversion_rate': 0.80, 'career_penalty_attempts': 40, 'recent_penalty_form': 0.75}
    ]
    
    team_b = [
        {'name': 'Enemy Star', 'player_penalty_conversion_rate': 0.80, 'career_penalty_attempts': 25, 'recent_penalty_form': 0.75},
        {'name': 'Enemy Mid', 'player_penalty_conversion_rate': 0.70, 'career_penalty_attempts': 12, 'recent_penalty_form': 0.65},
        {'name': 'Enemy Def', 'player_penalty_conversion_rate': 0.60, 'career_penalty_attempts': 4, 'recent_penalty_form': 0.55},
        {'name': 'Enemy Wing', 'player_penalty_conversion_rate': 0.72, 'career_penalty_attempts': 8, 'recent_penalty_form': 0.70},
        {'name': 'Enemy Vet', 'player_penalty_conversion_rate': 0.78, 'career_penalty_attempts': 35, 'recent_penalty_form': 0.72}
    ]
    
    n_simulations = 1000 
    print(f"Pre-computing probabilities for {n_simulations} shootouts (Vectorized)...")
    
    max_kicks = 15 
    team_a_probs = np.zeros((n_simulations, max_kicks))
    team_b_probs = np.zeros((n_simulations, max_kicks))
    
    for kick_idx in range(max_kicks):
        df_a = generate_random_kicks(team_a[kick_idx % 5], n_simulations)
        df_b = generate_random_kicks(team_b[kick_idx % 5], n_simulations)

        df_a_enc = pd.get_dummies(df_a)
        df_b_enc = pd.get_dummies(df_b)

        trained_features = striker_model.feature_names_
        df_a_aligned = df_a_enc.reindex(columns=trained_features, fill_value=0)
        df_b_aligned = df_b_enc.reindex(columns=trained_features, fill_value=0)
        
        # 1. Predict Agent Actions
        a_shot_pred = striker_model.predict(df_a_aligned)
        a_gk_pred = gk_model.predict(df_a_aligned)
        b_shot_pred = striker_model.predict(df_b_aligned)
        b_gk_pred = gk_model.predict(df_b_aligned)
        
        # 2. Evaluate Interaction
        df_a_outcome = df_a_aligned.copy()
        df_a_outcome['predicted_shot_dir'] = a_shot_pred
        df_a_outcome['predicted_gk_dive'] = a_gk_pred
        
        df_b_outcome = df_b_aligned.copy()
        df_b_outcome['predicted_shot_dir'] = b_shot_pred
        df_b_outcome['predicted_gk_dive'] = b_gk_pred
        
        # 3. Get Goal Probabilities
        team_a_probs[:, kick_idx] = outcome_model.predict_proba(df_a_outcome)[:, 1]
        team_b_probs[:, kick_idx] = outcome_model.predict_proba(df_b_outcome)[:, 1]
        
    print("Pre-computation complete. Running simulation loop...")
    
    team_a_wins = 0
    team_b_wins = 0
    
    for sim_idx in range(n_simulations):
        score_a, score_b = 0, 0
        
        for kick_idx in range(max_kicks):
            if np.random.rand() < team_a_probs[sim_idx, kick_idx]: score_a += 1
            if np.random.rand() < team_b_probs[sim_idx, kick_idx]: score_b += 1
                
            if kick_idx < 4:
                if score_a > score_b + (5 - (kick_idx + 1)): break
                if score_b > score_a + (5 - (kick_idx + 1)): break
            
        sudden_death_idx = 5
        while score_a == score_b and sudden_death_idx < max_kicks:
            if np.random.rand() < team_a_probs[sim_idx, sudden_death_idx]: score_a += 1
            if np.random.rand() < team_b_probs[sim_idx, sudden_death_idx]: score_b += 1
            sudden_death_idx += 1
            
        if score_a > score_b: team_a_wins += 1
        else: team_b_wins += 1
            
    print("\n--- Monte Carlo Simulation Results ---")
    print(f"Total Simulations: {n_simulations}")
    print(f"Team A Wins: {team_a_wins} ({(team_a_wins/n_simulations)*100:.1f}%)")
    print(f"Team B Wins: {team_b_wins} ({(team_b_wins/n_simulations)*100:.1f}%)")

def generate_random_kicks(shooter_profile, n):
    return pd.DataFrame({
        'player_id': np.random.randint(1, 5000, size=n),
        'player_penalty_conversion_rate': shooter_profile['player_penalty_conversion_rate'],
        'career_penalty_attempts': shooter_profile['career_penalty_attempts'],
        'recent_penalty_form': shooter_profile['recent_penalty_form'],
        'is_shootout': 1,
        'match_stage': 'Final',
        'minute_of_match': 120,
        'fatigue_index': np.random.normal(0.8, 0.1, size=n), # High fatigue in shootout
        'run_up_style': np.random.choice(['Straight', 'Curved', 'Stutter'], size=n),
        'pause_before_shot': np.random.choice([0, 1], size=n),
        'cv_body_lean_angle': np.random.normal(15, 5, size=n),
        'cv_run_up_speed': np.random.normal(8.5, 1.2, size=n),
        'goalkeeper_id': np.random.randint(5001, 8000, size=n),
        'goalkeeper_penalty_save_rate': 0.22,
        'goalkeeper_diving_bias': np.random.choice(['Left', 'Right', 'Center'], size=n),
        'keeper_experience_level': 4,
        'shooter_vs_keeper_history': 0,
        'psychological_advantage_index': 80
    })

if __name__ == "__main__":
    simulate_shootout()