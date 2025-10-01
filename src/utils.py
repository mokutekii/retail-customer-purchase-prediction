"""Misc utilities (placeholders)."""
from typing import Dict, Any

def set_seed(seed: int = 42):
    """Set random seeds across libs if needed."""
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
