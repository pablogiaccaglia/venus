"""Seed utilities for reproducibility."""

import os
import random
import numpy as np
import torch
from lightning.pytorch import seed_everything


def seed_worker(worker_id: int) -> None:
    """
    Worker initialization function for DataLoader reproducibility.
    
    Args:
        worker_id: Worker ID (unused but required by PyTorch)
    """
    worker_seed = 200
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_deterministic_mode(seed: int = 200) -> torch.Generator:
    """
    Set all random seeds and configure PyTorch for deterministic behavior.
    
    Args:
        seed: Random seed to use
    
    Returns:
        PyTorch Generator object with the seed set
    """
    print(f'Using random seed {seed}...')
    
    # Create generator
    g = torch.Generator()
    g.manual_seed(seed)
    
    # Set seeds everywhere
    seed_everything(seed, workers=True)
    
    # Environment variables for deterministic behavior
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Python random
    random.seed(seed)
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic algorithms (may impact performance)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    return g


def reseed() -> torch.Generator:
    """
    Reseed with the default seed value.
    
    Returns:
        PyTorch Generator object with the seed set
    """
    SEED = 200
    print(f'Using random seed {SEED}...')

    g = torch.Generator()
    g.manual_seed(SEED)

    seed_everything(SEED, workers=True)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(SEED)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    return g
