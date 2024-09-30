import torch.nn as nn
from types import SimpleNamespace
from einops import rearrange

class LeNet1D(nn.Module):
    def __init__(self, input_channels, input_length, num_classes=2, act_fn_name="relu"):
        super(LeNet1D, self).__init__()

        # Hyperparameters are stored in hparams for flexibility
        self.hparams = SimpleNamespace(
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            act_fn_name=act_fn_name,
            act_fn=self._get_activation_fn(act_fn_name)
        )
        
        # Create the network and initialize parameters
        self._create_network()
        self._init_params()

    def _create_network(self):
        """
        This method defines the layers for the LeNet1D model.
        """
        # Define the convolutional layers and pooling layers (body)
        self.body = nn.Sequential(
            # First convolution Layer (Conv1d)
            nn.Conv1d(in_channels=self.hparams.input_channels, out_channels=6, kernel_size=5),
            nn.BatchNorm1d(6),
            self.hparams.act_fn(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            # Second Convolution Layer
            nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5),
            nn.BatchNorm1d(16),
            self.hparams.act_fn(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )

        # Compute the output size after the convolutional layers
        conv_output_size = self._get_conv_output_size(self.hparams.input_length)

        # Define the fully connected layers (head)
        self.head = nn.Sequential(
            nn.Linear(in_features=16 * conv_output_size, out_features=120),
            self.hparams.act_fn(inplace=True),
            nn.Linear(in_features=120, out_features=84),
            self.hparams.act_fn(inplace=True),
            nn.Linear(in_features=84, out_features=self.hparams.num_classes)
        )

    def _init_params(self):
        """
        Initialize parameters of the model, especially convolutional layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # He initialization (kaiming) is good for ReLU activations
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.Linear):
                # Use a uniform initialization for linear layers
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _get_conv_output_size(self, input_length):
        """
        Compute the size of the output after the Conv1d and MaxPool1d layers.
        """
        output_length = (input_length - 4) // 2  # After first Conv1d and MaxPool1d
        output_length = (output_length - 4) // 2  # After second Conv1d and MaxPool1d
        return output_length

    def _get_activation_fn(self, act_fn_name):
        """
        Utility function to get the appropriate activation function by name.
        """
        if act_fn_name == "relu":
            return nn.ReLU
        elif act_fn_name == "tanh":
            return nn.Tanh
        elif act_fn_name == "leaky_relu":
            return nn.LeakyReLU
        else:
            raise ValueError(f"Unsupported activation function: {act_fn_name}")

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        # Pass the input tensor through the convolutional layers (body)
        x = self.body(x)
        # Flatten the output of the convolutional layers
        x = rearrange(x, 'b c l -> b (c l)')
        # Pass the flattened tensor through the fully connected layers (head)
        x = self.head(x)
        return x
