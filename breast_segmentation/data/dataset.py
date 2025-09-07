"""Dataset and data loading utilities for breast segmentation."""

import os
import copy
from typing import List, Tuple, Optional, Dict, Any
from natsort import natsorted, ns
import numpy as np
import monai
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose
import torch
from torch.utils.data import Dataset, default_collate

from ..config.settings import config


def get_image_label_files(
    dataset_base_path: str, split: str, image_type: str, dataset_name: str = "BREADM"
) -> Tuple[List[str], List[str]]:
    """
    Get lists of image and label file paths for a given dataset split.

    Args:
        dataset_base_path: Base path to the dataset
        split: Dataset split ('train', 'val', 'test')
        image_type: Type of images (e.g., 'SUB2')
        dataset_name: Name of the dataset ("BREADM" or "private")

    Returns:
        Tuple of (image_filenames, label_filenames)
    """
    all_images_fnames = []
    all_labels_fnames = []

    split_folder = os.path.join(dataset_base_path, split)

    # Get patient folders for images
    patients_images_folders_base_path = os.path.join(split_folder, "images")
    patients_images_folders = os.listdir(patients_images_folders_base_path)
    patients_images_folders = [
        os.path.join(patients_images_folders_base_path, p) for p in patients_images_folders
    ]
    patients_images_folders = natsorted(patients_images_folders, alg=ns.IGNORECASE)

    # Get patient folders for labels
    patients_labels_folders_base_path = os.path.join(split_folder, "labels")
    patients_labels_folders = os.listdir(patients_labels_folders_base_path)
    patients_labels_folders = [
        os.path.join(patients_labels_folders_base_path, p) for p in patients_labels_folders
    ]
    patients_labels_folders = natsorted(patients_labels_folders, alg=ns.IGNORECASE)

    # Iterate through patient folders
    for patient_images_folder, patient_labels_folder in zip(
        patients_images_folders, patients_labels_folders
    ):
        images_folder = os.path.join(patient_images_folder, image_type)
        labels_folder = os.path.join(patient_labels_folder, image_type)

        # Get image files
        images_fnames = os.listdir(images_folder)
        images_fnames = [os.path.join(images_folder, p) for p in images_fnames]
        images_fnames = natsorted(images_fnames, alg=ns.IGNORECASE)

        # Get label files
        labels_fnames = os.listdir(labels_folder)
        labels_fnames = [os.path.join(labels_folder, p) for p in labels_fnames]
        labels_fnames = natsorted(labels_fnames, alg=ns.IGNORECASE)

        all_images_fnames += images_fnames
        all_labels_fnames += labels_fnames

    return all_images_fnames, all_labels_fnames


def create_data_dicts(image_files: List[str], label_files: List[str]) -> List[Dict[str, str]]:
    """
    Create list of dictionaries for MONAI dataset.

    Args:
        image_files: List of image file paths
        label_files: List of label file paths

    Returns:
        List of dictionaries with 'image' and 'label' keys
    """
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(image_files, label_files)
    ]
    return data_dicts


def get_mean_std_dataloader(dataloader: DataLoader, masked: bool = False) -> Tuple[float, float]:
    """
    Calculate mean and standard deviation from a dataloader.

    Args:
        dataloader: DataLoader to calculate statistics from
        masked: Whether to use masked statistics (not implemented)

    Returns:
        Tuple of (mean, std)
    """
    # Variables to store sum and sum of squares
    sum_values = 0.0
    sum_of_squares = 0.0
    num_elements = 0

    # Iterate over the DataLoader
    for batch in dataloader:
        images = batch["image"]

        # Update the sum and sum of squares
        sum_values += images.sum().item()
        sum_of_squares += (images**2).sum().item()

        # Update the count of elements
        num_elements += images.numel()

    # Calculate the mean and standard deviation
    mean = sum_values / num_elements
    variance = (sum_of_squares / num_elements) - (mean**2)
    std = np.sqrt(variance)

    return mean, std


class PairedDataset(Dataset):
    """Dataset for paired training with two data sources and augmentation."""

    def __init__(
        self, dataset1: Dataset, dataset2: Dataset, augment: bool = False, filter_data: bool = False
    ):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.augmentations = Compose(
            [
                monai.transforms.RandHistogramShiftd(
                    keys=["image"], prob=0.2, num_control_points=4
                ),
                monai.transforms.RandRotated(
                    keys=["image", "label"], mode="nearest-exact", range_x=[0.1, 0.1], prob=0.3
                ),
                monai.transforms.RandZoomd(
                    keys=["image", "label"],
                    mode="nearest-exact",
                    min_zoom=1.3,
                    max_zoom=1.5,
                    prob=0.3,
                ),
            ]
        )
        self.augment = augment
        self.filter_data_samples = filter_data

    def __len__(self) -> int:
        return min(len(self.dataset1), len(self.dataset2))

    def filter_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter data based on keep_sample flag."""
        if not data["keep_sample"]:
            data["image"] = monai.data.MetaTensor(torch.zeros_like(data["image"]))
        return data

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        data1 = self.dataset1[idx]
        data2 = self.dataset2[idx]

        # Extract the two patches from dataset2

        if self.filter_data_samples:
            data2_flat = [self.filter_data(item) for item in data2]
            data2_1 = data2_flat[0]
            data2_2 = data2_flat[1]

        else:
            data2_1 = data2[0]
            data2_2 = data2[1]

        if self.augment:
            self.augmentations.set_random_state(seed=idx)

            data2_1 = {
                "image": copy.deepcopy(data2_1["image"]),
                "label": copy.deepcopy(data2_1["label"]),
                "boundary": copy.deepcopy(data2_1.get("boundary", np.array([]))),
            }
            data2_2 = {
                "image": copy.deepcopy(data2_2["image"]),
                "label": copy.deepcopy(data2_2["label"]),
                "boundary": copy.deepcopy(data2_2.get("boundary", np.array([]))),
            }
            data1 = {
                "image": copy.deepcopy(data1["image"]),
                "label": copy.deepcopy(data1["label"]),
                "boundary": copy.deepcopy(data1.get("boundary", np.array([]))),
            }

            data2_1 = self.augmentations(data2_1)

            self.augmentations.set_random_state(seed=idx)
            data2_2 = self.augmentations(data2_2)

            self.augmentations.set_random_state(seed=idx)
            data1 = self.augmentations(data1)

        return data1, data2_1, data2_2


class PairedDataLoader(DataLoader):
    """DataLoader for paired datasets."""

    def __init__(
        self,
        dataset1: Dataset,
        dataset2: Dataset,
        batch_size: int,
        shuffle: bool,
        worker_init_fn: Optional[Any],
        generator: Optional[torch.Generator],
        drop_last: bool,
        num_workers: Optional[int] = None,
        augment: bool = False,
    ):
        paired_dataset = PairedDataset(dataset1, dataset2, augment=augment)
        super().__init__(
            paired_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            worker_init_fn=worker_init_fn,
            generator=generator,
            drop_last=drop_last,
        )
