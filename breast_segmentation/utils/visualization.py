"""
Visualization utilities for breast segmentation models.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple
from monai.data import CacheDataset
from breast_segmentation.models import BreastFusionModel, BreastSegmentationModel
from breast_segmentation.utils.inference import get_image_label_files_patient_aware, reverse_transformations


def plot_slices_side_by_side(images: List[np.ndarray], titles: List[str] = None, 
                           figsize: Tuple[int, int] = (15, 5), cmap: str = 'gray'):
    """Plot multiple slices side by side."""
    if titles is None:
        titles = [f'Slice {i}' for i in range(len(images))]
    
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    if len(images) == 1:
        axes = [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap=cmap)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5,
                  mask_color: Tuple[float, float, float] = (1.0, 0.0, 0.0)):
    """Create an overlay of image and mask."""
    if image.ndim == 2:
        image_rgb = np.stack([image, image, image], axis=-1)
    else:
        image_rgb = image.copy()
    
    # Normalize image to [0, 1]
    image_rgb = (image_rgb - image_rgb.min()) / (image_rgb.max() - image_rgb.min())
    
    # Apply mask overlay
    mask_indices = mask > 0
    for i, color_val in enumerate(mask_color):
        image_rgb[mask_indices, i] = alpha * color_val + (1 - alpha) * image_rgb[mask_indices, i]
    
    return image_rgb


def visualize_predictions(
    model_path: str,
    patient_id: str,
    dataset_key: str,
    dataset_base_path: str,
    data_transforms: Dict[str, Any],
    slice_idx: int = 0,
    model_type: str = 'venus',
    arch_name: Optional[str] = None,
    base_channels: int = 64,
    use_simple_fusion: bool = True,
    use_decoder_attention: bool = True,
    threshold: float = 0.4,
    num_workers: int = 0,
    figsize: tuple = (15, 10),
    image_type: str = "VIBRANT+C2",
    split: str = 'test'
):
    """
    Visualize predictions for a specific patient and slice.
    
    Args:
        model_path: Path to the model checkpoint
        patient_id: Patient ID to visualize
        dataset_key: Dataset key ('no_thorax_sub_test_ds' or 'no_thorax_sub_thorax_test_ds')
        dataset_base_path: Base path to the dataset
        data_transforms: Dictionary mapping dataset keys to transforms
        slice_idx: Slice index to visualize
        model_type: 'venus' or 'baseline'
        arch_name: Architecture name for baseline models
        base_channels: Base channels for VENUS models
        use_simple_fusion: Whether to use simple fusion (VENUS models)
        use_decoder_attention: Whether to use decoder attention (VENUS models)
        threshold: Threshold for binary prediction
        num_workers: Number of workers for data loading
        figsize: Figure size for visualization
    """
    
    patient_images, patient_labels = get_image_label_files_patient_aware(
        dataset_base_path,
        split,
        image_type,
        patient_id
    )
    
    if slice_idx >= len(patient_images):
        print(f"Slice index {slice_idx} out of range for patient {patient_id} (max: {len(patient_images)-1})")
        return
    
    # Create dataset for single slice
    slice_data = [{"image": patient_images[slice_idx], "label": patient_labels[slice_idx]}]
    slice_dataset = CacheDataset(
        data=slice_data,
        transform=data_transforms[dataset_key],
        cache_rate=1.0,
        num_workers=num_workers
    )
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == 'venus':
        model = BreastFusionModel.load_from_checkpoint(
            model_path,
            use_simple_fusion=use_simple_fusion,
            use_decoder_attention=use_decoder_attention,
            arch=None,
            strict=False,
            base_channels=base_channels,
            map_location=device
        )
    else:
        model = BreastSegmentationModel.load_from_checkpoint(
            model_path,
            arch_name=arch_name,
            strict=False,
            map_location=device
        )
    
    model.eval()
    model.to(device)
    
    # Get prediction
    with torch.no_grad():
        batch = slice_dataset[0]
        image = batch["image"].unsqueeze(0).to(device)
        label = batch["label"].unsqueeze(0).to(device)
        
        # Get prediction
        if hasattr(model, 'ttaug') and hasattr(model.ttaug, '__call__'):
            pred = model.ttaug(image)
        else:
            pred = model(image)
        
        # Apply sigmoid if needed
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Move to CPU for visualization
        image = image.cpu().numpy()[0, 0]
        label = label.cpu().numpy()[0, 0]
        pred_prob = pred.cpu().numpy()[0, 0]
        pred_binary = (pred_prob > threshold).astype(np.float32)
    
    pred_reversed, label_reversed, image_reversed = reverse_transformations(
        torch.tensor(pred_prob).unsqueeze(0).unsqueeze(0),
        torch.tensor(label).unsqueeze(0).unsqueeze(0),
        torch.tensor(image).unsqueeze(0).unsqueeze(0),
        batch
    )
    
    # Convert back to numpy
    pred_reversed = pred_reversed.squeeze().numpy()
    label_reversed = label_reversed.squeeze().numpy()
    image_reversed = image_reversed.squeeze().numpy()
    pred_binary_reversed = (pred_reversed > threshold).astype(np.float32)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Top row - transformed space
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title(f'Input Image (Transformed)\nPatient: {patient_id}, Slice: {slice_idx}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(label, cmap='gray')
    axes[0, 1].set_title('Ground Truth (Transformed)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_binary, cmap='gray')
    axes[0, 2].set_title(f'Prediction (Transformed)\nThreshold: {threshold}')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(image_reversed, cmap='gray')
    axes[1, 0].set_title('Input Image (Original)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(label_reversed, cmap='gray')
    axes[1, 1].set_title('Ground Truth (Original)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(pred_binary_reversed, cmap='gray')
    axes[1, 2].set_title('Prediction (Original)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and display metrics for this slice
    tp = np.sum((pred_binary_reversed > 0) & (label_reversed > 0))
    fp = np.sum((pred_binary_reversed > 0) & (label_reversed == 0))
    fn = np.sum((pred_binary_reversed == 0) & (label_reversed > 0))
    tn = np.sum((pred_binary_reversed == 0) & (label_reversed == 0))
    
    if tp + fp + fn > 0:
        dice = 2 * tp / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\nSlice Metrics:")
        print(f"Dice: {dice:.4f}")
        print(f"IoU: {iou:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
    else:
        print("\nNo positive predictions or ground truth in this slice.")


def visualize_volume_predictions(
    model_path: str,
    patient_id: str,
    dataset_key: str,
    dataset_base_path: str,
    data_transforms: Dict[str, Any],
    model_type: str = 'venus',
    arch_name: Optional[str] = None,
    base_channels: int = 64,
    use_simple_fusion: bool = True,
    use_decoder_attention: bool = True,
    threshold: float = 0.4,
    num_workers: int = 0,
    max_slices: int = 9,
    figsize: tuple = (15, 10),
    split: str = 'test'
):
    """
    Visualize predictions for multiple slices of a patient volume.
    
    Args:
        model_path: Path to the model checkpoint
        patient_id: Patient ID to visualize
        dataset_key: Dataset key ('no_thorax_sub_test_ds' or 'no_thorax_sub_thorax_test_ds')
        dataset_base_path: Base path to the dataset
        data_transforms: Dictionary mapping dataset keys to transforms
        model_type: 'venus' or 'baseline'
        arch_name: Architecture name for baseline models
        base_channels: Base channels for VENUS models
        use_simple_fusion: Whether to use simple fusion (VENUS models)
        use_decoder_attention: Whether to use decoder attention (VENUS models)
        threshold: Threshold for binary prediction
        num_workers: Number of workers for data loading
        max_slices: Maximum number of slices to visualize
        figsize: Figure size for visualization
    """
    # Get patient files
    patient_images, patient_labels = get_image_label_files_patient_aware(
        dataset_base_path,
        split,
        "VIBRANT+C2",  # Use the same image type as in visualize_predictions
        patient_id
    )
    
    num_slices = min(len(patient_images), max_slices)
    cols = 3
    rows = (num_slices + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Load model once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == 'venus':
        model = BreastFusionModel.load_from_checkpoint(
            model_path,
            use_simple_fusion=use_simple_fusion,
            use_decoder_attention=use_decoder_attention,
            arch=None,
            strict=False,
            base_channels=base_channels,
            map_location=device
        )
    else:
        model = BreastSegmentationModel.load_from_checkpoint(
            model_path,
            arch_name=arch_name,
            strict=False,
            map_location=device
        )
    
    model.eval()
    model.to(device)
    
    for i in range(num_slices):
        row = i // cols
        col = i % cols
        
        # Create dataset for single slice
        slice_data = [{"image": patient_images[i], "label": patient_labels[i]}]
        slice_dataset = CacheDataset(
            data=slice_data,
            transform=data_transforms[dataset_key],
            cache_rate=1.0,
            num_workers=num_workers
        )
        
        # Get prediction
        with torch.no_grad():
            batch = slice_dataset[0]
            image = batch["image"].unsqueeze(0).to(device)
            label = batch["label"].unsqueeze(0).to(device)
            
            # Get prediction
            if hasattr(model, 'ttaug') and hasattr(model.ttaug, '__call__'):
                pred = model.ttaug(image)
            else:
                pred = model(image)
            
            # Apply sigmoid if needed
            if pred.min() < 0 or pred.max() > 1:
                pred = torch.sigmoid(pred)
            
            pred_prob = pred.cpu().numpy()[0, 0]
            pred_binary = (pred_prob > threshold).astype(np.float32)
        
        # Reverse transformations
        pred_reversed, label_reversed, image_reversed = reverse_transformations(
            torch.tensor(pred_prob).unsqueeze(0).unsqueeze(0),
            torch.tensor(label.cpu().numpy()[0, 0]).unsqueeze(0).unsqueeze(0),
            torch.tensor(image.cpu().numpy()[0, 0]).unsqueeze(0).unsqueeze(0),
            batch
        )
        
        pred_binary_reversed = (pred_reversed.squeeze().numpy() > threshold).astype(np.float32)
        label_reversed = label_reversed.squeeze().numpy()
        image_reversed = image_reversed.squeeze().numpy()
        
        # Create overlay
        overlay = image_reversed.copy()
        overlay = np.stack([overlay, overlay, overlay], axis=-1)
        overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min())
        
        # Add ground truth in green and prediction in red
        overlay[label_reversed > 0, 1] = 1.0  # Ground truth in green
        overlay[pred_binary_reversed > 0, 0] = 1.0  # Prediction in red
        
        axes[row, col].imshow(overlay)
        axes[row, col].set_title(f'Slice {i}\nGT=Green, Pred=Red')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(num_slices, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Volume Predictions for {patient_id}', fontsize=16)
    plt.tight_layout()
    plt.show()