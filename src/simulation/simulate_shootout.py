import pandas as pd
import os
import joblib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "calibrated_penalty_model.joblib")

def simulate_shootout():
    print("Loading ML Model for simulation...")
    model = joblib.load(MODEL_PATH)
    
    # 1. Define two teams (5 shooters each)
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
    
    n_simulations = 10000
    print(f"Pre-computing probabilities for {n_simulations} shootouts (Vectorized)...")
    
    max_kicks = 15 
    
    team_a_probs = np.zeros((n_simulations, max_kicks))
    team_b_probs = np.zeros((n_simulations, max_kicks))
    
    for kick_idx in range(max_kicks):
        shooter_a = team_a[kick_idx % 5]
        shooter_b = team_b[kick_idx % 5]
        
        df_a = generate_random_kicks(shooter_a, n_simulations)
        df_b = generate_random_kicks(shooter_b, n_simulations)
        
        df_a_enc = pd.get_dummies(df_a)
        df_b_enc = pd.get_dummies(df_b)
        
        trained_features = model.calibrated_classifiers_[0].estimator.feature_names_
        df_a_aligned = df_a_enc.reindex(columns=trained_features, fill_value=0)
        df_b_aligned = df_b_enc.reindex(columns=trained_features, fill_value=0)
        
        team_a_probs[:, kick_idx] = model.predict_proba(df_a_aligned)[:, 1]
        team_b_probs[:, kick_idx] = model.predict_proba(df_b_aligned)[:, 1]
        
    print("Pre-computation complete. Running simulation loop...")
    
    team_a_wins = 0
    team_b_wins = 0
    
    for sim_idx in range(n_simulations):
        score_a = 0
        score_b = 0
        
        for kick_idx in range(max_kicks):
            if np.random.rand() < team_a_probs[sim_idx, kick_idx]:
                score_a += 1
                
            if np.random.rand() < team_b_probs[sim_idx, kick_idx]:
                score_b += 1
                
            remaining_kicks = 5 - (kick_idx + 1)
            if kick_idx >= 4: 
                remaining_kicks = 0
                
            if kick_idx < 4:
                if score_a > score_b + (5 - (kick_idx + 1)):
                    break
                if score_b > score_a + (5 - (kick_idx + 1)):
                    break
            
        sudden_death_idx = 5
        while score_a == score_b and sudden_death_idx < max_kicks:
            if np.random.rand() < team_a_probs[sim_idx, sudden_death_idx]:
                score_a += 1
            if np.random.rand() < team_b_probs[sim_idx, sudden_death_idx]:
                score_b += 1
            sudden_death_idx += 1
            
        if score_a > score_b:
            team_a_wins += 1
        else:
            team_b_wins += 1
            
    print("\n--- Monte Carlo Simulation Results ---")
    print(f"Total Simulations: {n_simulations}")
    print(f"Team A Wins: {team_a_wins} ({(team_a_wins/n_simulations)*100:.1f}%)")
    print(f"Team B Wins: {team_b_wins} ({(team_b_wins/n_simulations)*100:.1f}%)")

def generate_random_kicks(shooter_profile, n):
    shot_dir = np.random.choice(['Left', 'Right', 'Center'], size=n)
    gk_dir = np.random.choice(['Left', 'Right', 'Center'], size=n)
    gk_correct = (shot_dir == gk_dir).astype(int)
    
    return pd.DataFrame({
        'player_penalty_conversion_rate': shooter_profile['player_penalty_conversion_rate'],
        'career_penalty_attempts': shooter_profile['career_penalty_attempts'],
        'recent_penalty_form': shooter_profile['recent_penalty_form'],
        'is_shootout': 1,
        'match_stage': 'Final',
        'minute_of_match': 120,
        'team_pressure_state': 'Drawing',
        'run_up_style': np.random.choice(['Straight', 'Curved', 'Stutter'], size=n),
        'pause_before_shot': np.random.choice([0, 1], size=n),
        'shot_telegraph_level': np.random.choice([1, 2, 3], size=n),
        'goalkeeper_penalty_save_rate': 0.22,
        'goalkeeper_diving_bias': np.random.choice(['Left', 'Right', 'Center'], size=n),
        'keeper_experience_level': 4,
        'shot_direction': shot_dir,
        'shot_height': np.random.choice(['Low', 'Middle', 'High'], size=n),
        'corner_precision': np.random.normal(0.8, 0.1, size=n),
        'goalkeeper_dive_direction': gk_dir,
        'gk_correct_guess': gk_correct,
        'reaction_time_ms': np.random.normal(220, 30, size=n),
        'shooter_vs_keeper_history': 0,
        'psychological_advantage_index': 80
    })

if __name__ == "__main__":
    simulate_shootout()