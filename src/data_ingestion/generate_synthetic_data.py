import pandas as pd
import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic")

def main():
    print("Generating PIS² Dataset (50k rows, 28 features)...")
    n_samples = 50000
    np.random.seed(42) 

    # 1-4. Shooter Skill & History
    player_id = np.random.randint(1, 5000, n_samples)
    player_penalty_conversion_rate = np.random.normal(0.75, 0.1, n_samples)
    career_penalty_attempts = np.random.poisson(20, n_samples)
    recent_penalty_form = np.random.normal(0.7, 0.15, n_samples)

    # 5-8. Match Context & Pressure
    is_shootout = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    match_stage = np.random.choice(['Group', 'Knockout', 'Final'], size=n_samples, p=[0.6, 0.3, 0.1])
    minute_of_match = np.random.randint(1, 121, n_samples)
    fatigue_index = (minute_of_match / 120.0) * np.random.normal(0.5, 0.1, n_samples)

    # 9-12. Biomechanics & CV Features
    run_up_style = np.random.choice(['Straight', 'Curved', 'Stutter'], size=n_samples, p=[0.4, 0.4, 0.2])
    pause_before_shot = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    cv_body_lean_angle = np.random.normal(15, 5, n_samples)
    cv_run_up_speed = np.random.normal(8.5, 1.2, n_samples)

    # 13-16. Goalkeeper Profile
    goalkeeper_id = np.random.randint(5001, 8000, n_samples)
    goalkeeper_penalty_save_rate = np.random.normal(0.22, 0.05, n_samples)
    goalkeeper_diving_bias = np.random.choice(['Left', 'Right', 'Center'], size=n_samples, p=[0.45, 0.45, 0.10])
    keeper_experience_level = np.random.choice([1, 2, 3, 4, 5], size=n_samples)

    # 17-20. Shot Mechanics
    shot_direction = np.random.choice(['Left', 'Center', 'Right'], size=n_samples, p=[0.4, 0.1, 0.5])
    shot_height = np.random.choice(['Low', 'Middle', 'High'], size=n_samples, p=[0.6, 0.1, 0.3])
    corner_precision = np.random.normal(0.8, 0.15, n_samples)
    shot_speed_kmh = np.random.normal(85, 10, n_samples)

    # 21-22. Duel & Game Theory
    shooter_vs_keeper_history = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    psychological_advantage_index = np.random.randint(0, 100, n_samples)

    # 23-28. Deep Learning Embeddings (Simulated Vector Outputs)
    embed_cols = {f'embed_{i}': np.random.normal(0, 1, n_samples) for i in range(6)}

    gk_dive_direction = []
    synth_goals = []
    
    for i in range(n_samples):
        # GK decides to dive based on bias or history
        if np.random.rand() < 0.6:
            dive = goalkeeper_diving_bias[i]
        else:
            dive = np.random.choice(['Left', 'Right', 'Center'])
        gk_dive_direction.append(dive)
        
        prob = player_penalty_conversion_rate[i]
        if is_shootout[i] == 1: prob -= 0.05
        if match_stage[i] == 'Final': prob -= 0.05
        if fatigue_index[i] > 0.7: prob -= 0.05
        
        if dive == shot_direction[i]:
            prob -= 0.40
            if shot_height[i] == 'Low': prob -= 0.05
            if shot_speed_kmh[i] < 75: prob -= 0.10
        else:
            prob += 0.20
            
        prob = max(0.05, min(0.98, prob))
        synth_goals.append(np.random.choice([1, 0], p=[prob, 1-prob]))

    df = pd.DataFrame({
        'player_id': player_id, 'player_penalty_conversion_rate': player_penalty_conversion_rate,
        'career_penalty_attempts': career_penalty_attempts, 'recent_penalty_form': recent_penalty_form,
        'is_shootout': is_shootout, 'match_stage': match_stage, 'minute_of_match': minute_of_match,
        'fatigue_index': fatigue_index, 'run_up_style': run_up_style, 'pause_before_shot': pause_before_shot,
        'cv_body_lean_angle': cv_body_lean_angle, 'cv_run_up_speed': cv_run_up_speed,
        'goalkeeper_id': goalkeeper_id, 'goalkeeper_penalty_save_rate': goalkeeper_penalty_save_rate,
        'goalkeeper_diving_bias': goalkeeper_diving_bias, 'keeper_experience_level': keeper_experience_level,
        'shot_direction': shot_direction, 'shot_height': shot_height, 'corner_precision': corner_precision,
        'shot_speed_kmh': shot_speed_kmh, 'shooter_vs_keeper_history': shooter_vs_keeper_history,
        'psychological_advantage_index': psychological_advantage_index, **embed_cols,
        'gk_dive_direction': gk_dive_direction, # NEW TARGET FOR GK MODEL
        'is_goal': synth_goals
    })
    
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    output_path = os.path.join(SYNTHETIC_DIR, "synthetic_penalties_50k.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ PIS² Dataset saved to: {output_path}")

if __name__ == "__main__":
    main()