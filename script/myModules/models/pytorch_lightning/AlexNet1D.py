import torch.nn as nn
from types import SimpleNamespace
from einops import rearrange
import torch

class AlexNet1D(nn.Module):
    def __init__(self, input_channels, input_length, num_classes=2, act_fn_name="relu"):
        super(AlexNet1D, self).__init__()

        self.hparams = SimpleNamespace(
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            act_fn_name=act_fn_name,
            act_fn=self._get_activation_fn(act_fn_name)
        )

        self.body = nn.Sequential(
            nn.Conv1d(in_channels=self.hparams.input_channels, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm1d(64),
            self.hparams.act_fn(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Conv1d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm1d(192),
            self.hparams.act_fn(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Conv1d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384),
            self.hparams.act_fn(inplace=True),

            nn.Conv1d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            self.hparams.act_fn(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            self.hparams.act_fn(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool1d(6)
        )

        # Dynamically compute the output size of the conv layers
        self._conv_output_size = self._get_conv_output(input_channels, input_length)

        # Define the fully connected layers (head)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=self._conv_output_size, out_features=1024),
            self.hparams.act_fn(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=1024),
            self.hparams.act_fn(inplace=True),
            nn.Linear(in_features=1024, out_features=self.hparams.num_classes)
        )

    def _get_activation_fn(self, act_fn_name):
        if act_fn_name == "relu":
            return nn.ReLU
        elif act_fn_name == "tanh":
            return nn.Tanh
        elif act_fn_name == "leaky_relu":
            return nn.LeakyReLU
        else:
            raise ValueError(f"Unsupported activation function: {act_fn_name}")

    def _get_conv_output(self, input_channels, input_length):
        # Create a dummy input to pass through the conv layers
        dummy_input = torch.randn(1, input_channels, input_length)
        output_feat = self.body(dummy_input)
        return output_feat.numel()  # Get the total number of elements
    
    def forward(self, x):
        x = self.body(x)
        x = rearrange(x, 'b c l -> b (c l)')
        x = self.head(x)
        return x
