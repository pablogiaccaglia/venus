"""Configuration settings for the breast segmentation pipeline."""

import os
from typing import Optional


class Config:
    """Main configuration class for the breast segmentation pipeline."""

    # Random seed for reproducibility
    SEED: int = 200

    # Data split ratios
    TRAIN_RATIO: float = 0.8
    VALIDATION_RATIO: float = 0.4
    TEST_RATIO: float = 0.2  # 1 - TRAIN_RATIO

    # Training settings
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = os.cpu_count()

    # Paths
    checkpoints_dir_breadm: str = "./checkpoints/breadm-dataset"
    checkpoints_dir_private: str = "./checkpoints/private-dataset"

    DATASET_BASE_PATH_BREADM: str = "BreaDM/seg"
    DATASET_BASE_PATH_PRIVATE: str = "Dataset-arrays-4-FINAL"

    MAX_EPOCHS: int = 1000
    EARLY_STOPPING_PATIENCE: int = 10
    GRADIENT_CLIP_VAL: int = 1
    LOG_EVERY_N_STEPS: int = 10

    # Image settings
    IMAGE_SIZE: tuple = (256, 256)
    IN_CHANNELS: int = 1
    OUT_CHANNELS: int = 1  

    # Preprocessing parameters (BREADM dataset)
    MEDIAN_SMOOTH_RADIUS: int = 2
    HIST_NORM_NUM_BINS: int = 40
    RM_THORAX_THRESHOLD: float = 80
    RM_THORAX_MARGIN: int = 80
    RM_BOTTOM_THRESHOLD: int = 80
    RM_BOTTOM_MARGIN: int = 50
    THRESHOLD_BLACK_THRESHOLD: float = 1300
    THRESHOLD_BLACK_VALUE: float = 0
    TRIM_SIDES_THRESHOLD: float = 20000
    TRIM_SIDES_TOLERANCE: int = 20

    # Private dataset preprocessing parameters
    PRIVATE_HIST_NORM_NUM_BINS: int = 20
    PRIVATE_RM_THORAX_THRESHOLD: float = 150
    PRIVATE_RM_THORAX_MARGIN: int = 60
    PRIVATE_RM_BOTTOM_THRESHOLD: int = 120
    PRIVATE_RM_BOTTOM_MARGIN: int = 50

    TARGET_SIZE_PRELIMINARY: tuple = (512, 512)
    TARGET_SIZE_FINAL: tuple = (256, 256)
    BBOX_SIZE: tuple = (384, 384)
    PAD_SPATIAL_SIZE: list = [512, 512]

    # Conditional parameters based on patches
    MEAN_THRESHOLD_PATCHES: float = 50
    START_POS_PATCHES: int = 75
    MEAN_THRESHOLD_NO_PATCHES: float = 35
    START_POS_NO_PATCHES: int = 40

    # Environment variables for deterministic behavior
    CUDA_LAUNCH_BLOCKING: str = "1"
    TORCH_USE_CUDA_DSA: str = "1"
    CUBLAS_WORKSPACE_CONFIG: str = ":4096:8"

    def __init__(self):
        """Initialize configuration and set environment variables."""
        self._setup_environment()

    def _setup_environment(self):
        """Set up environment variables for deterministic behavior."""
        os.environ["CUDA_LAUNCH_BLOCKING"] = self.CUDA_LAUNCH_BLOCKING
        os.environ["TORCH_USE_CUDA_DSA"] = self.TORCH_USE_CUDA_DSA
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = self.CUBLAS_WORKSPACE_CONFIG
        os.environ["PYTHONHASHSEED"] = str(self.SEED)


# Create a global config instance
config = Config()
