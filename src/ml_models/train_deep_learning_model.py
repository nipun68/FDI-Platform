import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic", "synthetic_penalties_50k.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

class PenaltyDNN(nn.Module):
    def __init__(self, num_players, num_gks, embed_dim=16):
        super(PenaltyDNN, self).__init__()
        self.player_embed = nn.Embedding(num_players, embed_dim)
        self.gk_embed = nn.Embedding(num_gks, embed_dim)
        self.fc1 = nn.Linear(embed_dim * 2 + 20, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, player_id, gk_id, features):
        p_emb = self.player_embed(player_id).squeeze(1)
        g_emb = self.gk_embed(gk_id).squeeze(1)

        x = torch.cat([p_emb, g_emb, features], dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def main():
    print("Loading data for Deep Learning Model (PyTorch)...")
    df = pd.read_csv(DATA_PATH)

    df['player_id'] = df['player_id'] - df['player_id'].min()
    df['goalkeeper_id'] = df['goalkeeper_id'] - df['goalkeeper_id'].min()
    
    num_players = df['player_id'].max() + 1
    num_gks = df['goalkeeper_id'].max() + 1

    num_features = [
        'player_penalty_conversion_rate', 'career_penalty_attempts', 'recent_penalty_form', 
        'is_shootout', 'minute_of_match', 'fatigue_index', 'pause_before_shot', 
        'cv_body_lean_angle', 'cv_run_up_speed', 'goalkeeper_penalty_save_rate', 
        'keeper_experience_level', 'shooter_vs_keeper_history', 
        'psychological_advantage_index', 'corner_precision', 'shot_speed_kmh',
        'embed_0', 'embed_1', 'embed_2', 'embed_3', 'embed_4'
    ]
    
    scaler = StandardScaler()
    df[num_features] = scaler.fit_transform(df[num_features])
    
    X_train, X_test, y_train, y_test = train_test_split(df, df['is_goal'], test_size=0.2, random_state=42)

    def get_tensors(df_split):
        p_ids = torch.tensor(df_split['player_id'].values, dtype=torch.long).unsqueeze(1)
        g_ids = torch.tensor(df_split['goalkeeper_id'].values, dtype=torch.long).unsqueeze(1)
        feats = torch.tensor(df_split[num_features].values, dtype=torch.float32)
        targets = torch.tensor(df_split['is_goal'].values, dtype=torch.float32).unsqueeze(1)
        return p_ids, g_ids, feats, targets

    p_train, g_train, f_train, y_train_t = get_tensors(X_train)
    p_test, g_test, f_test, y_test_t = get_tensors(X_test)
    
    # Initialize Model, Loss, Optimizer
    model = PenaltyDNN(num_players, num_gks)
    criterion = nn.BCELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training Deep Learning Model...")
    epochs = 15
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(p_train, g_train, f_train)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(p_test, g_test, f_test).numpy()
        
    print(f"\n--- Deep Learning Model Evaluation ---")
    print(f"Brier Score: {brier_score_loss(y_test, preds):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, preds):.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "deep_learning_model.pt"))
    print("✅ Deep Learning Model saved as 'deep_learning_model.pt'")

if __name__ == "__main__":
    main()