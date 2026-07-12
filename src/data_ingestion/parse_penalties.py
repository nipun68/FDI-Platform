import json
import os
import glob
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
EVENTS_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "statsbomb", "events")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

def extract_penalties():
    """Reads all raw event files and extracts penalty data into a flat DataFrame."""
    
    all_penalties = []
    
    event_files = glob.glob(os.path.join(EVENTS_DIR, "*.json"))
    print(f"Found {len(event_files)} match files to process...")

    for file_path in event_files:
        match_id = os.path.basename(file_path).replace('.json', '')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            events = json.load(f)
            
        for event in events:
            
            if event.get('type', {}).get('name') == 'Shot':
                if event.get('shot', {}).get('type', {}).get('name') == 'Penalty':
                    
                    
                    penalty_data = {
                        'match_id': match_id,
                        'period': event.get('period'),
                        'minute': event.get('minute'),
                        'player_name': event.get('player', {}).get('name'),
                        'team_name': event.get('team', {}).get('name'),
                        
                        # Outcome: 1 if Goal, 0 if Missed/Saved
                        'is_goal': 1 if event.get('shot', {}).get('outcome', {}).get('name') == 'Goal' else 0,
                        'outcome_name': event.get('shot', {}).get('outcome', {}).get('name'),
                    }
                    
                    # Extract location (x, y) - StatsBomb uses [0-120] for x, [0-80] for y
                    location = event.get('location')
                    if location:
                        penalty_data['start_x'] = location[0]
                        penalty_data['start_y'] = location[1]
                        
                    
                    end_location = event.get('shot', {}).get('end_location')
                    if end_location:
                        penalty_data['end_x'] = end_location[0] # Usually 120 (goal line)
                        if len(end_location) > 1:
                            penalty_data['end_y'] = end_location[1] # Y coordinate (0-80)
                        if len(end_location) > 2:
                            penalty_data['end_z'] = end_location[2] # Z coordinate (height)
                            
                    
                    if 'goalkeeper' in event.get('shot', {}):
                        penalty_data['gk_name'] = event['shot']['goalkeeper'].get('player', {}).get('name')
                        penalty_data['gk_position'] = event['shot']['goalkeeper'].get('position', {}).get('name')
                    else:
                        penalty_data['gk_name'] = None
                        penalty_data['gk_position'] = None

                    all_penalties.append(penalty_data)

    
    df = pd.DataFrame(all_penalties)
    return df

def main():
    print("Extracting penalty data...")
    df = extract_penalties()
    
    if df.empty:
        print("No penalties found!")
        return
        
    print(f"Extracted {len(df)} penalties.")
    
    
    print("\n--- Summary ---")
    print(f"Total Penalties: {len(df)}")
    print(f"Goals Scored: {df['is_goal'].sum()}")
    print(f"Conversion Rate: {df['is_goal'].mean() * 100:.1f}%")
    
   
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, "penalties_raw.csv")
    df.to_csv(output_path, index=False)
    print(f"\nData saved to: {output_path}")

if __name__ == "__main__":
    main()