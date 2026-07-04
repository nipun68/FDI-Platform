import pandas as pd
import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
REAL_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "features", "penalties_features.csv")
SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic")

def main():
    print("Loading real World Cup data to learn distributions...")
    df_real = pd.read_csv(REAL_DATA_PATH)
    
    real_goal_rate = df_real['is_goal'].mean()
    real_dist_mean, real_dist_std = df_real['shot_distance'].mean(), df_real['shot_distance'].std()
    real_angle_mean, real_angle_std = df_real['shot_angle'].mean(), df_real['shot_angle'].std()
    
    print(f"Real Goal Rate: {real_goal_rate*100:.1f}%")
    print(f"Real Distance: {real_dist_mean:.2f}m (+/- {real_dist_std:.2f})")
    
    n_samples = 5000
    print(f"\nGenerating {n_samples} synthetic penalties...")
    
    np.random.seed(42) 

    synth_distance = np.random.normal(11.0, 0.2, n_samples) 
    synth_angle = np.random.normal(22.0, 1.0, n_samples)   
    synth_shootout = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]) 
    
    synth_goals = []
    for dist, angle, shootout in zip(synth_distance, synth_angle, synth_shootout):     
        prob = 0.75
        if shootout == 1:
            prob -= 0.10

        prob -= (dist - 11.0) * 0.01
        prob -= (angle - 22.0) * 0.01

        prob = max(0.10, min(0.95, prob))
   
        goal = np.random.choice([1, 0], p=[prob, 1-prob])
        synth_goals.append(goal)

    df_synth = pd.DataFrame({
        'player_name': 'Synthetic_Player',
        'shot_distance': synth_distance,
        'shot_angle': synth_angle,
        'is_shootout': synth_shootout,
        'is_goal': synth_goals
    })

    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    output_path = os.path.join(SYNTHETIC_DIR, "synthetic_penalties_5k.csv")
    df_synth.to_csv(output_path, index=False)
    
    print(f"\n✅ Synthetic data saved to: {output_path}")
    print(f"Synthetic Goal Rate: {df_synth['is_goal'].mean()*100:.1f}%")
    print(f"Synthetic Shootout Goal Rate: {df_synth[df_synth['is_shootout']==1]['is_goal'].mean()*100:.1f}%")

if __name__ == "__main__":
    main()