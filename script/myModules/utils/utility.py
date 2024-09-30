import subprocess
import numpy as np
import torch


def cuda_device_count() -> int:
    """
    Get the number of NVIDIA GPUs available in the environment.

    Returns:
        (int): The number of NVIDIA GPUs available.
    """
    try:
        # Run the nvidia-smi command and capture its output
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"], encoding="utf-8"
        )

        # Take the first line and strip any leading/trailing white space
        first_line = output.strip().split("\n")[0]

        return int(first_line)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        # If the command fails, nvidia-smi is not found, or output is not an integer, assume no GPUs are available
        return 0

def cuda_is_available() -> bool:
    """
    Check if CUDA is available in the environment.

    Returns:
        (bool): True if one or more NVIDIA GPUs are available, False otherwise.
    """
    return cuda_device_count() > 0

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)