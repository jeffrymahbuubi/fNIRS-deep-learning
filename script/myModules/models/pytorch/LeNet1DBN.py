import torch.nn as nn

class LeNet1DBN(nn.Module):
    def __init__(self, input_channels, input_length, num_classes=2):
        super(LeNet1DBN, self).__init__()

        # convolutional layers
        self._body = nn.Sequential(
            # First convolution Layer (Conv1d)
            # input shape = (batch_size, input_channels, input_length)
            nn.Conv1d(in_channels=input_channels, out_channels=6, kernel_size=5),
            nn.BatchNorm1d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            # Second Convolution Layer
            nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )

        # Compute the output size after the convolutional layers and pooling layers
        conv_output_size = self._get_conv_output_size(input_length)

        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=16 * conv_output_size, out_features=120),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=84, out_features=num_classes)
        )

    def _get_conv_output_size(self, input_length):
        # Compute the size of the output after two Conv1d and MaxPool1d layers
        output_length = (input_length - 4) // 2 # After first Conv1d and MaxPool1d
        output_length = (output_length - 4) // 2 # After second Conv1d and MaxPool1d
        return output_length
    
    def forward(self, x):
        # Pass the input tensor through the convolutional layers
        x = self._body(x)
        # Flatten the output of the convolutional layers
        x = x.view(x.size()[0], -1)
        # Pass the flattened tensor through the fully connected layers
        x = self.head(x)
        return x