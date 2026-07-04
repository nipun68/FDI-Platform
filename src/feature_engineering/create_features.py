import pandas as pd
import numpy as np
import os
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "penalties_raw.csv")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")

PITCH_LENGTH = 120
PITCH_WIDTH = 80
GOAL_WIDTH = 8 # Standard goal is 8 yards/meters wide

def calculate_distance(x, y):
    """Calculates distance from ball start position to the center of the goal."""
    goal_x = PITCH_LENGTH
    goal_y = PITCH_WIDTH / 2
    return math.sqrt((x - goal_x)**2 + (y - goal_y)**2)

def calculate_angle(x, y):
    """Calculates the shot angle in degrees using the law of cosines."""
    goal_x = PITCH_LENGTH
    goal_left_y = (PITCH_WIDTH / 2) - (GOAL_WIDTH / 2)
    goal_right_y = (PITCH_WIDTH / 2) + (GOAL_WIDTH / 2)

    dist_left = math.sqrt((x - goal_x)**2 + (y - goal_left_y)**2)
    dist_right = math.sqrt((x - goal_x)**2 + (y - goal_right_y)**2)
    
    # Law of cosines to find the angle at the ball
    # c^2 = a^2 + b^2 - 2ab*cos(C) -> cos(C) = (a^2 + b^2 - c^2) / 2ab
    # where a=dist_left, b=dist_right, c=GOAL_WIDTH
    if dist_left == 0 or dist_right == 0:
        return 0.0
        
    cos_angle = (dist_left**2 + dist_right**2 - GOAL_WIDTH**2) / (2 * dist_left * dist_right)

    cos_angle = max(min(cos_angle, 1), -1)
    
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)

def categorize_placement(y, z):
    """Categorizes shot placement into 6 zones (Left, Center, Right -> High/Low)."""
   
    if 36 <= y <= 44:
        horizontal = "Center"
    elif y < 36:
        horizontal = "Left"
    else:
        horizontal = "Right"

    if z is None or pd.isna(z):
        return f"{horizontal} (Unknown)"
    
    if z < 2.5:
        vertical = "Low"
    else:
        vertical = "High"
    return f"{vertical} {horizontal}"

def main():
    print("Loading raw penalties data...")
    if not os.path.exists(RAW_DATA_PATH):
        print("Error: penalties_raw.csv not found. Run parse_penalties.py first!")
        return
        
    df = pd.read_csv(RAW_DATA_PATH)
    
    print("Engineering features...")
    df['shot_distance'] = df.apply(lambda row: calculate_distance(row['start_x'], row['start_y']), axis=1)
    df['shot_angle'] = df.apply(lambda row: calculate_angle(row['start_x'], row['start_y']), axis=1)
    df['placement_zone'] = df.apply(lambda row: categorize_placement(row['end_y'], row['end_z']), axis=1)
    df['is_shootout'] = df['period'].apply(lambda x: 1 if x == 5 else 0)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    output_path = os.path.join(FEATURES_DIR, "penalties_features.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\n[SUCCESS] Feature engineering complete! Saved to {output_path}")
    print("\n--- Data Preview ---")
    print(df[['player_name', 'is_goal', 'shot_distance', 'shot_angle', 'placement_zone', 'is_shootout']].head())

if __name__ == "__main__":
    main()