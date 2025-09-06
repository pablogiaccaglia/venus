"""
Custom collate functions for data loading in breast segmentation training.

These functions handle filtering, augmentation, and batch collation for different
dataset configurations and training modes.
"""

import copy
import numpy as np
import torch
from torch.utils.data import default_collate
from monai.transforms import Compose
import monai.transforms


def custom_collate(batch):
    """
    Standard custom collate function for patches-based datasets.
    Filters out samples that should not be kept based on 'keep_sample' flag.
    Handles nested batch structure from patch datasets.
    
    Args:
        batch: Batch of data where each item may be a list of patches
        
    Returns:
        Collated batch or None if no valid samples
    """
    # Filter out None samples and flatten nested structure
    batch = [item for sublist in batch for item in sublist if item['keep_sample']]
    
    if len(batch) > 0:
        batch = default_collate(batch)
        return batch
    return None


def custom_collate_no_patches(batch):
    """
    Custom collate function for non-patches datasets.
    Filters out samples that should not be kept based on 'keep_sample' flag.
    
    Args:
        batch: Batch of data items
        
    Returns:
        Collated batch or None if no valid samples
    """
    # Filter out None samples
    batch = [item for item in batch if item['keep_sample']]
    
    if len(batch) > 0:
        batch = default_collate(batch)
        return batch
    return None