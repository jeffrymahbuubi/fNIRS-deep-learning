from einops import rearrange
import torch.nn as nn

class LeNet2D(nn.Module):
    """
    LeNet2D model for MNIST Dataset.
    This class implements the LeNet architecture using 2D convolutional layers.
    The model consists of two main parts: the convolutional layers (feature extractor)
    and the fully connected layers (classifier).
    Attributes:
        _body (nn.Sequential): Sequential container for convolutional layers.
        _head (nn.Sequential): Sequential container for fully connected layers.
    Methods:
        forward(X):
            Defines the forward pass of the model.
            Args:
                X (torch.Tensor): Input tensor of shape (batch_size, 1, 32, 32).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, 10).
    """
    def __init__(self):
        super().__init__()

        # convolutional layers
        self._body = nn.Sequential(
            # First Convolution layer
            # input size = (32, 32), output size = (28, 28)
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            # Relu activation
            nn.ReLU(inplace=True),
            # Max Pooling 2-d
            nn.MaxPool2d(kernel_size=2),

            # Second Convolution layer
            # input size = (14, 14), output_size = (10, 10)
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
            # output size = (5, 5)
        )

        # Fully connected layers
        self._head = nn.Sequential(
            # First Fully connected layer 
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            
            # ReLU activation
            nn.ReLU(inplace=True),

            # Second Fully connected layer
            nn.Linear(in_features=120, out_features=84),

            # ReLU activation
            nn.ReLU(inplace=True),

            # Third Fully connected layer
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, X):
        # apply feature extractor
        x = self._body(X)
        # flatten the output of conv layers using einops
        x = rearrange(x, 'b c h w -> b (c h w)')
        # apply classification head
        x = self._head(x)
        return x
