import pandas as pd
import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic")

def main():
    print("Generating ULTRA-REALISTIC synthetic penalties (7 Feature Categories)...")
    
    n_samples = 10000 
    np.random.seed(42) 

    player_penalty_conversion_rate = np.random.normal(0.75, 0.1, n_samples)
    career_penalty_attempts = np.random.poisson(20, n_samples)
    recent_penalty_form = np.random.normal(0.7, 0.15, n_samples)

    is_shootout = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    match_stage = np.random.choice(['Group', 'Knockout', 'Final'], size=n_samples, p=[0.6, 0.3, 0.1])
    minute_of_match = np.random.randint(1, 121, n_samples)
    team_pressure_state = np.random.choice(['Winning', 'Losing', 'Drawing'], size=n_samples, p=[0.3, 0.3, 0.4])

    run_up_style = np.random.choice(['Straight', 'Curved', 'Stutter'], size=n_samples, p=[0.4, 0.4, 0.2])
    pause_before_shot = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    shot_telegraph_level = np.random.choice([1, 2, 3], size=n_samples, p=[0.3, 0.4, 0.3]) # 3 = very predictable
    
    goalkeeper_penalty_save_rate = np.random.normal(0.22, 0.05, n_samples)
    goalkeeper_diving_bias = np.random.choice(['Left', 'Right', 'Center'], size=n_samples, p=[0.45, 0.45, 0.10])
    keeper_experience_level = np.random.choice([1, 2, 3, 4, 5], size=n_samples)
    
    shot_direction = np.random.choice(['Left', 'Center', 'Right'], size=n_samples, p=[0.4, 0.1, 0.5])
    shot_height = np.random.choice(['Low', 'Middle', 'High'], size=n_samples, p=[0.6, 0.1, 0.3])
    corner_precision = np.random.normal(0.8, 0.15, n_samples) # 1.0 = right in the top corner

    # If shooter stutters or telegraphs heavily, GK is more likely to guess the correct side
    gk_correct_guess = np.zeros(n_samples)
    for i in range(n_samples):
        prob_correct = 0.30 # Base 30% chance GK guesses right
        if shot_telegraph_level[i] == 3: prob_correct += 0.20
        if run_up_style[i] == 'Stutter' and pause_before_shot[i] == 0: prob_correct += 0.10 # Bad stutter
        
        # Check if dive matches shot direction
        if np.random.rand() < prob_correct:
            goalkeeper_dive_direction = shot_direction[i]
            gk_correct_guess[i] = 1
        else:
            # Dive wrong way
            choices = ['Left', 'Right', 'Center']
            choices.remove(shot_direction[i])
            goalkeeper_dive_direction = np.random.choice(choices)
            
    reaction_time_ms = np.random.normal(220, 40, n_samples) - (keeper_experience_level * 10)
    
    # Interaction & Psychology
    shooter_vs_keeper_history = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]) # Rare they've faced each other
    psychological_advantage_index = np.random.randint(0, 100, n_samples)
    
    
    synth_goals = []
    for i in range(n_samples):
        prob = player_penalty_conversion_rate[i]
        
        # Pressure effects
        if is_shootout[i] == 1: prob -= 0.05
        if match_stage[i] == 'Final': prob -= 0.05
        if team_pressure_state[i] == 'Losing': prob -= 0.05
        
        # Placement vs GK Duel
        if gk_correct_guess[i] == 1:
            prob -= 0.40 # GK guessed right!
            if shot_height[i] == 'Low': prob -= 0.05 # Easier to save low if you guess right
            if corner_precision[i] > 0.9: prob += 0.15 # Unsaveable even if guessed right
            if reaction_time_ms[i] < 180: prob -= 0.10 # GK reacted fast
        else:
            prob += 0.20 # GK dove wrong way
            
        # Telegraphing
        if shot_telegraph_level[i] == 3 and gk_correct_guess[i] == 0:
            prob -= 0.05 # Telegraphed but GK still guessed wrong? Likely a bad shot.
            
        prob = max(0.05, min(0.98, prob))
        goal = np.random.choice([1, 0], p=[prob, 1-prob])
        synth_goals.append(goal)
        
    df = pd.DataFrame({
        'player_penalty_conversion_rate': player_penalty_conversion_rate,
        'career_penalty_attempts': career_penalty_attempts,
        'recent_penalty_form': recent_penalty_form,
        'is_shootout': is_shootout,
        'match_stage': match_stage,
        'minute_of_match': minute_of_match,
        'team_pressure_state': team_pressure_state,
        'run_up_style': run_up_style,
        'pause_before_shot': pause_before_shot,
        'shot_telegraph_level': shot_telegraph_level,
        'goalkeeper_penalty_save_rate': goalkeeper_penalty_save_rate,
        'goalkeeper_diving_bias': goalkeeper_diving_bias,
        'keeper_experience_level': keeper_experience_level,
        'shot_direction': shot_direction,
        'shot_height': shot_height,
        'corner_precision': corner_precision,
        'goalkeeper_dive_direction': goalkeeper_dive_direction,
        'gk_correct_guess': gk_correct_guess,
        'reaction_time_ms': reaction_time_ms,
        'shooter_vs_keeper_history': shooter_vs_keeper_history,
        'psychological_advantage_index': psychological_advantage_index,
        'is_goal': synth_goals
    })
    
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    output_path = os.path.join(SYNTHETIC_DIR, "synthetic_penalties_10k.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Ultra-Realistic data saved to: {output_path}")
    print(f"Total Features: {len(df.columns) - 1}")
    print(f"Overall Goal Rate: {df['is_goal'].mean()*100:.1f}%")

if __name__ == "__main__":
    main()