import torch.nn as nn
from einops import rearrange

class LSTM1DClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=2):
        super(LSTM1DClassifier, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 84),  # Fully connected layer with 84 units
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)  # Output layer (binary classification)
        )

    def forward(self, x):
        # Reshape from (batch_size, channels, time_steps) to (batch_size, time_steps, channels)
        x = rearrange(x, 'b c t -> b t c')

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, time_steps, hidden_size)

        # Take the output from the last time step (last output of the LSTM)
        out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)

        return self.fc(out)
