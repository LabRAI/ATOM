"""
utils/seed_utils.py

This script provides a utility function to set random seeds for reproducibility in machine learning experiments.
It ensures consistent results by setting seeds for:
- PyTorch (CPU and CUDA)
- NumPy
- Python's random module
- Torch Geometric (if used)

Additionally, it configures PyTorch's backend to ensure deterministic behavior.
"""

import torch
import numpy as np
import random

from torch_geometric.seed import seed_everything

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seed_everything(seed)
