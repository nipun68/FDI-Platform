import requests
import json
import os
import time

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "statsbomb")

def fetch_data(url):
    """Helper function to fetch JSON data from a URL."""
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code} from {url}")
        return None

def save_json(data, filepath):
    """Helper function to save JSON data to a local file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def download_world_cup_2022():
    """Downloads all matches and events for the 2022 FIFA World Cup."""
    

    os.makedirs(DATA_DIR, exist_ok=True)
    
 
    print("Fetching competitions list...")
    comps = fetch_data(BASE_URL + "competitions.json")
    if not comps:
        return
    
  
    wc_2022 = next((c for c in comps if c.get("competition_id") == 43 and c.get("season_id") == 106), None)
    if not wc_2022:
        print("Could not find 2022 World Cup data.")
        return
        
    print(f"Found competition: {wc_2022['competition_name']} - {wc_2022['season_name']}")
    
    
    matches_url = f"{BASE_URL}matches/{wc_2022['competition_id']}/{wc_2022['season_id']}.json"
    print("Fetching matches list...")
    matches = fetch_data(matches_url)
    if not matches:
        return
        
    print(f"Found {len(matches)} matches. Starting download...")
    
   
    for i, match in enumerate(matches):
        match_id = match['match_id']
        home_team = match['home_team']['home_team_name']
        away_team = match['away_team']['away_team_name']
        
        match_filename = os.path.join(DATA_DIR, f"matches", f"{match_id}.json")
        save_json(match, match_filename)
        
        events_url = f"{BASE_URL}events/{match_id}.json"
        print(f"[{i+1}/{len(matches)}] Downloading events for: {home_team} vs {away_team} (ID: {match_id})...")
        
        events = fetch_data(events_url)
        if events:
            events_filename = os.path.join(DATA_DIR, f"events", f"{match_id}.json")
            save_json(events, events_filename)
            
       
        time.sleep(0.5) 

    print("\nDownload complete! All World Cup 2022 data saved to /data/raw/statsbomb/")

if __name__ == "__main__":
    download_world_cup_2022()