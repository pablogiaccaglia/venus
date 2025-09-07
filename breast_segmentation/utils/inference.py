"""Utilities for inference and post-processing."""

import torch
import numpy as np
import monai
from typing import Dict, Optional, Tuple
import torch.nn.functional as F


def reverse_transformations(d, processed_label, mode="patches"):
    # Extract the processed label and transformation coordinates
    y1_crop, y2_crop, x1_crop, x2_crop = d["crop_coords"]
    x1_bottom, y1_bottom, x2_bottom, y2_bottom = d["bottom_crop_coords"]

    if mode == "patches":
        start_breast, end_breast = d["trim_breast_coords"]
    x1_thorax, y1_thorax, x2_thorax, y2_thorax = d["thorax_crop_coords"]
    intermediate_spatial_dim = d["dim_before_resize_final"]

    label_resized = (
        F.interpolate(
            processed_label.unsqueeze(0).unsqueeze(0).float(),
            size=intermediate_spatial_dim.tolist(),  # Excluding the batch size dimension
            mode="nearest-exact",
        )
        .squeeze(0)
        .squeeze(0)
    )  # Removing the added batch and channel dimensions

    pad_post_crop_coords = d["pad_post_crop_coords"].tolist()
    before1, before2, before3 = (
        pad_post_crop_coords[0],
        pad_post_crop_coords[1],
        pad_post_crop_coords[2],
    )

    original_height = label_resized.shape[1] - before2[1]
    original_width = label_resized.shape[2] - before3[1]

    # Slice the image to remove the padding
    reversed_pad = label_resized[:, :original_height, :original_width]

    # Step 1: Reverse final crop
    crop_height, crop_width = d["dim_before_crop"][1:]
    padded_label = torch.zeros((1, crop_height, crop_width), dtype=label_resized.dtype)
    padded_label[:, y1_crop:y2_crop, x1_crop:x2_crop] = reversed_pad

    # Step 2: Reverse bottom crop
    bottom_height = d["dim_before_bottom_crop"][1]
    bottom_width = d["dim_before_bottom_crop"][2]
    bottom_padded_label = torch.zeros((1, bottom_height, x2_bottom), dtype=padded_label.dtype)
    bottom_padded_label[:, :y2_bottom, :x2_bottom] = padded_label

    # Conditional steps based on whether breast trim was applied
    if mode == "patches":
        # Step 3: Reverse breast trim
        trim_width = d["dim_before_breast_crop"][2]
        trim_padded_label = torch.zeros(
            (bottom_height, trim_width), dtype=bottom_padded_label.dtype
        )
        trim_padded_label[:, start_breast:end_breast] = bottom_padded_label
    else:
        # In other modes, use the bottom_padded_label directly for thorax crop reversal
        trim_padded_label = bottom_padded_label
        trim_width = bottom_width  # This assumes no trimming, hence the width is unchanged

    # Step 4: Reverse thorax crop
    thorax_height = d["dim_before_thorax_crop"][1]
    thorax_padded_label = torch.zeros((1, thorax_height, trim_width), dtype=trim_padded_label.dtype)
    thorax_padded_label[:, y1_thorax:, :] = trim_padded_label

    original_spatial_dim = d["dim_before_resize_preliminary"]

    reconstructed_mask = (
        F.interpolate(
            thorax_padded_label.unsqueeze(0).unsqueeze(0).float(),
            size=original_spatial_dim.tolist(),  # Excluding the batch size dimension
            mode="nearest-exact",
        )
        .squeeze(0)
        .squeeze(0)
    )  # Removing the added batch and channel dimensions

    return reconstructed_mask


def get_patient_ids(dataset_base_path: str, split: str) -> list:
    """
    Get patient IDs from dataset structure.

    Args:
        dataset_base_path: Base path to dataset
        split: Dataset split ('train', 'val', 'test')

    Returns:
        List of patient IDs
    """
    import os

    split_folder = os.path.join(dataset_base_path, split)
    patients_images_folders_base_path = os.path.join(split_folder, "images")
    patient_ids = os.listdir(patients_images_folders_base_path)
    return patient_ids


def get_image_label_files_patient_aware(
    dataset_base_path: str, split: str, image_type: str, patient_id: str
) -> Tuple[list, list]:
    """
    Get image and label files for a specific patient.

    Args:
        dataset_base_path: Base path to dataset
        split: Dataset split
        image_type: Type of images
        patient_id: Patient ID

    Returns:
        Tuple of (image_files, label_files)
    """
    import os
    from natsort import natsorted, ns

    split_folder = os.path.join(dataset_base_path, split)

    patients_images_folders_base_path = os.path.join(split_folder, "images")
    patients_images_folders = os.path.join(
        patients_images_folders_base_path, patient_id + "/" + image_type
    )

    patients_labels_folders_base_path = os.path.join(split_folder, "labels")
    patients_labels_folders = os.path.join(
        patients_labels_folders_base_path, patient_id + "/" + image_type
    )

    images_fnames = os.listdir(patients_images_folders)
    images_fnames = [os.path.join(patients_images_folders, p) for p in images_fnames]
    images_fnames = natsorted(images_fnames, alg=ns.IGNORECASE)

    labels_fnames = os.listdir(patients_labels_folders)
    labels_fnames = [os.path.join(patients_labels_folders, p) for p in labels_fnames]
    labels_fnames = natsorted(labels_fnames, alg=ns.IGNORECASE)

    return images_fnames, labels_fnames
