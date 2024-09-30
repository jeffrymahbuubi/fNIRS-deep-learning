"""
The :mod:`pytorch` module provides a collection of PyTorch-based models that can be used for various tasks within this project, primarily focused on deep learning-based signal processing, image processing, and video processing.

Author: Aunuun Jeffry Mahbuubi (National Cheng Kung University, BME, WTMH Lab, 2024)

Changelog:
Version 1.0.0:
- Create the module

Version: 1.0.0
"""

from .LeNet1D import LeNet1D
from .AlexNet1D import AlexNet1D
from .TCN import TCNForClassification as TCN

__all__ = ["LeNet1D", "AlexNet1D", "TCN"]