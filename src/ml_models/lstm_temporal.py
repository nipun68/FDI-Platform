import torch
import torch.nn as nn

class PenaltyLSTM(nn.Module):
    """LSTM model for analyzing temporal patterns in a player's penalty history."""
    def __init__(self, input_size=5, hidden_size=16, num_layers=2):
        super(PenaltyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Take the last time step
        return self.sigmoid(out)

# Example usage:
# model = PenaltyLSTM()
# sequence = torch.randn(1, 10, 5) # 10 past penalties, 5 features each
# future_form = model(sequence)