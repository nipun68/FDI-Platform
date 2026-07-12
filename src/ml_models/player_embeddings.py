import torch
import torch.nn as nn

class PlayerTransformerEmbedder(nn.Module):
    """Generates deep contextual embeddings for players based on their history."""
    def __init__(self, num_players=5000, embed_dim=6):
        super(PlayerTransformerEmbedder, self).__init__()
        # Embedding layer maps player IDs to dense vectors
        self.embedding = nn.Embedding(num_embeddings=num_players, embedding_dim=embed_dim)
        # Transformer processes the sequence
        self.transformer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=2, batch_first=True)

    def forward(self, player_ids):
        """Returns a 6-dimensional vector representing player style/skill."""
        x = self.embedding(player_ids)
        # Transformer processes the sequence (even if length 1 for a single kick)
        out = self.transformer(x) 
        return out.mean(dim=1) # Pool to single vector

# Example usage for production:
# model = PlayerTransformerEmbedder()
# player_id_tensor = torch.tensor([101]) 
# embedding_vector = model(player_id_tensor)
# print(embedding_vector.shape) # Output: torch.Size([1, 6])