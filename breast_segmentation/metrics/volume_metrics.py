"""Volume-based filtering and metrics for 3D medical images."""

import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import label as labell, find_objects, generate_binary_structure
import torch
import cv2


def filter_masses(volume, min_slices=7, window_size=3):
    """
    Filters out masses in a 3D volume (HxWxB) that do not consecutively appear in at least 'min_slices' slices,
    considering a window around each mass. This function uses cv2 for dilation and assumes binary masks as input.

    :param volume: 3D numpy array of shape (H, W, B) representing a volume of binary masks.
    :param min_slices: Minimum number of consecutive slices a mass must appear in to be kept.
    :param window_size: Diameter of the window used for dilation to connect varying shapes.
    :return: Filtered 3D volume.
    """
    volume_copy = np.copy(volume)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))

    for i in range(volume_copy.shape[2]):
        volume_copy[:, :, i] = cv2.dilate(
            volume_copy[:, :, i].astype(np.uint8), kernel, iterations=1
        )

    volume_copy_transposed = np.transpose(volume_copy, (2, 0, 1))
    volume_transposed = np.transpose(volume, (2, 0, 1))

    structure = generate_binary_structure(3, 1)
    labeled_volume, num_features = labell(volume_copy_transposed, structure=structure)
    print(f"num features: {num_features}")
    for feature_id in range(1, num_features + 1):
        feature_mask = labeled_volume == feature_id
        slice_presence_count = np.sum(np.any(feature_mask, axis=(1, 2)))

        if slice_presence_count < min_slices:
            volume_transposed[labeled_volume == feature_id] = 0  # Filter out the feature

    filtered_volume = np.transpose(volume_transposed, (1, 2, 0))
    print("fine")
    return filtered_volume


def remove_far_masses_based_on_largest_mass(
    predicted_volume: np.ndarray, distance_threshold: float = 50.0
) -> np.ndarray:
    """
    Remove masses that are too far from the largest mass.
    Implementation from reference notebook.

    Args:
        predicted_volume: 3D binary volume (H x W x N)
        distance_threshold: Maximum distance from largest mass

    Returns:
        Filtered 3D binary volume with distant masses removed
    """
    # Label connected components
    labeled_volume, num_labels = label(predicted_volume)

    if num_labels == 0:
        return predicted_volume

    # Find the largest component
    component_sizes = []
    component_centroids = []

    for label_id in range(1, num_labels + 1):
        component_mask = labeled_volume == label_id
        component_size = np.sum(component_mask)
        component_sizes.append(component_size)

        # Calculate centroid
        coords = np.where(component_mask)
        centroid = np.array([np.mean(coords[i]) for i in range(3)])
        component_centroids.append(centroid)

    # Find largest component
    largest_idx = np.argmax(component_sizes)
    largest_centroid = component_centroids[largest_idx]

    # Filter components based on distance
    filtered_volume = np.zeros_like(predicted_volume)

    for idx, centroid in enumerate(component_centroids):
        distance = np.linalg.norm(centroid - largest_centroid)

        if distance <= distance_threshold:
            component_mask = labeled_volume == idx + 1
            filtered_volume[component_mask] = 1

    return filtered_volume


def compute_iou_npy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_id: int = 1,
    exclude_empty: bool = False,
    reduction: str = "micro",
) -> float:
    """
    Compute IoU score for numpy arrays.

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        class_id: Class ID to compute IoU for
        exclude_empty: Whether to exclude empty cases
        reduction: Reduction method ('micro', 'micro-imagewise')

    Returns:
        IoU score
    """
    if class_id == 1:
        gt_mask = y_true > 0
        pred_mask = y_pred > 0
    else:
        gt_mask = y_true == 0
        pred_mask = y_pred == 0

    if reduction == "micro":
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()

        if union == 0:
            return 1.0 if not exclude_empty else 0.0

        return intersection / union

    elif reduction == "micro-imagewise":
        # Compute IoU per slice, then average
        if len(gt_mask.shape) == 3:  # Volume with multiple slices
            ious = []
            for i in range(gt_mask.shape[2]):  # Iterate over slices
                gt_slice = gt_mask[:, :, i]
                pred_slice = pred_mask[:, :, i]

                intersection = np.logical_and(gt_slice, pred_slice).sum()
                union = np.logical_or(gt_slice, pred_slice).sum()

                if union == 0:
                    if not exclude_empty:
                        ious.append(1.0)
                    # Skip empty slices if exclude_empty=True
                else:
                    ious.append(intersection / union)

            return np.mean(ious) if ious else 0.0
        else:
            # Single slice
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()

            if union == 0:
                return 1.0 if not exclude_empty else 0.0

            return intersection / union

    else:
        raise ValueError(f"Unsupported reduction method: {reduction}")


def compute_dice_score_npy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_id: int = 1,
    exclude_empty: bool = False,
    reduction: str = "micro",
) -> float:
    """
    Compute Dice score for numpy arrays.

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        class_id: Class ID to compute Dice for
        exclude_empty: Whether to exclude empty cases
        reduction: Reduction method ('micro', 'micro-imagewise')

    Returns:
        Dice score
    """
    if class_id == 1:
        gt_mask = y_true > 0
        pred_mask = y_pred > 0
    else:
        gt_mask = y_true == 0
        pred_mask = y_pred == 0

    if reduction == "micro":
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        total = gt_mask.sum() + pred_mask.sum()

        if total == 0:
            return 1.0 if not exclude_empty else 0.0

        return (2.0 * intersection) / total

    elif reduction == "micro-imagewise":
        # Compute Dice per slice, then average
        if len(gt_mask.shape) == 3:  # Volume with multiple slices
            dices = []
            for i in range(gt_mask.shape[2]):  # Iterate over slices
                gt_slice = gt_mask[:, :, i]
                pred_slice = pred_mask[:, :, i]

                intersection = np.logical_and(gt_slice, pred_slice).sum()
                total = gt_slice.sum() + pred_slice.sum()

                if total == 0:
                    if not exclude_empty:
                        dices.append(1.0)
                    # Skip empty slices if exclude_empty=True
                else:
                    dices.append((2.0 * intersection) / total)

            return np.mean(dices) if dices else 0.0
        else:
            # Single slice
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            total = gt_mask.sum() + pred_mask.sum()

            if total == 0:
                return 1.0 if not exclude_empty else 0.0

            return (2.0 * intersection) / total

    else:
        raise ValueError(f"Unsupported reduction method: {reduction}")


def compute_accuracy_from_cumulator(
    TPs, FPs, FNs, TNs, exclude_empty=False, is_mean=True, return_std=False
):
    """Compute accuracy from confusion matrix components."""
    import torch

    tp = torch.cat([tp for tp in TPs])
    fp = torch.cat([fp for fp in FPs])
    fn = torch.cat([fn for fn in FNs])
    tn = torch.cat([tn for tn in TNs])

    # Denominators for accuracy
    denom_class_1 = tp + fp + fn
    denom_class_0 = tn + fp + fn

    # Accuracy for class 1 (foreground)
    accuracy_class_1 = torch.where(
        denom_class_1 > 0,
        tp / denom_class_1,
        torch.tensor(1.0 if not exclude_empty else float("nan")),
    )
    mean_accuracy_class_1 = torch.nanmean(accuracy_class_1).item()
    stddev_class_1 = np.nanstd(accuracy_class_1.cpu().numpy())

    # Accuracy for class 0 (background)
    accuracy_class_0 = torch.where(
        denom_class_0 > 0,
        tn / denom_class_0,
        torch.tensor(1.0 if not exclude_empty else float("nan")),
    )
    mean_accuracy_class_0 = torch.nanmean(accuracy_class_0).item()
    stddev_class_0 = np.nanstd(accuracy_class_0.cpu().numpy())

    if not is_mean:
        if return_std:
            # Return per-class accuracy and standard deviations
            return mean_accuracy_class_1, stddev_class_1
        else:
            # Return per-class accuracy only
            return mean_accuracy_class_1

    # Overall mean accuracy (mean between the two classes)
    overall_mean_accuracy = (mean_accuracy_class_1 + mean_accuracy_class_0) / 2

    overall_std_accuracy = (stddev_class_1 + stddev_class_0) / 2

    if return_std:
        # Return overall mean accuracy and standard deviations
        return overall_mean_accuracy, overall_std_accuracy
    else:
        # Return overall mean accuracy (single value when is_mean=True)
        return overall_mean_accuracy


def compute_mean_precision_from_cumulator(tp, fp, fn, tn) -> float:
    """Compute mean precision from confusion matrix components."""
    # Convert to numpy arrays if they are lists
    tp = np.array(tp) if isinstance(tp, list) else tp
    fp = np.array(fp) if isinstance(fp, list) else fp
    fn = np.array(fn) if isinstance(fn, list) else fn
    tn = np.array(tn) if isinstance(tn, list) else tn

    # Precision for class 1
    prec_1 = tp.sum() / (tp.sum() + fp.sum()) if (tp.sum() + fp.sum()) > 0 else 0.0
    # Precision for class 0
    prec_0 = tn.sum() / (tn.sum() + fn.sum()) if (tn.sum() + fn.sum()) > 0 else 0.0
    return (prec_1 + prec_0) / 2.0


def compute_mean_recall_from_cumulator(tp, fp, fn, tn) -> float:
    """Compute mean recall from confusion matrix components."""
    # Convert to numpy arrays if they are lists
    tp = np.array(tp) if isinstance(tp, list) else tp
    fp = np.array(fp) if isinstance(fp, list) else fp
    fn = np.array(fn) if isinstance(fn, list) else fn
    tn = np.array(tn) if isinstance(tn, list) else tn

    # Recall for class 1
    rec_1 = tp.sum() / (tp.sum() + fn.sum()) if (tp.sum() + fn.sum()) > 0 else 0.0
    # Recall for class 0
    rec_0 = tn.sum() / (tn.sum() + fp.sum()) if (tn.sum() + fp.sum()) > 0 else 0.0
    return (rec_1 + rec_0) / 2.0


def compute_precision_from_cumulator(
    TPs, FPs, FNs, TNs, exclude_empty=False, is_mean=True, return_std=False
):
    """Compute precision from confusion matrix components."""
    import torch

    tp = torch.cat([tp for tp in TPs])
    fp = torch.cat([fp for fp in FPs])
    fn = torch.cat([fn for fn in FNs])
    tn = torch.cat([tn for tn in TNs])

    # Denominators for precision
    denom_class_1 = tp + fp
    denom_class_0 = tn + fn

    # Precision for class 1 (per sample)
    precision_class_1 = torch.where(
        denom_class_1 > 0,
        tp / denom_class_1,
        torch.tensor(1.0 if not exclude_empty else float("nan")),
    )
    mean_precision_class_1 = torch.nanmean(precision_class_1).item()
    stddev_class_1 = np.nanstd(precision_class_1.cpu().numpy())

    # Precision for class 0 (per sample)
    precision_class_0 = torch.where(
        denom_class_0 > 0,
        tn / denom_class_0,
        torch.tensor(1.0 if not exclude_empty else float("nan")),
    )
    mean_precision_class_0 = torch.nanmean(precision_class_0).item()
    stddev_class_0 = np.nanstd(precision_class_0.cpu().numpy())

    if not is_mean:
        if return_std:
            return mean_precision_class_1, stddev_class_1
        else:
            return mean_precision_class_1

    # Overall mean precision (mean between the two classes)
    overall_mean_precision = (mean_precision_class_1 + mean_precision_class_0) / 2

    overall_std_precision = (stddev_class_1 + stddev_class_0) / 2

    if return_std:
        return overall_mean_precision, overall_std_precision
    else:
        return overall_mean_precision


def compute_recall_from_cumulator(
    TPs, FPs, FNs, TNs, exclude_empty=False, is_mean=True, return_std=False
):
    """Compute recall from confusion matrix components."""
    import torch

    tp = torch.cat([tp for tp in TPs])
    fp = torch.cat([fp for fp in FPs])
    fn = torch.cat([fn for fn in FNs])
    tn = torch.cat([tn for tn in TNs])

    # Denominators for recall
    denom_class_1 = tp + fn
    denom_class_0 = tn + fp

    # Recall for class 1 (per sample)
    recall_class_1 = torch.where(
        denom_class_1 > 0,
        tp / denom_class_1,
        torch.tensor(1.0 if not exclude_empty else float("nan")),
    )
    mean_recall_class_1 = torch.nanmean(recall_class_1).item()
    stddev_class_1 = np.nanstd(recall_class_1.cpu().numpy())

    # Recall for class 0 (per sample)
    recall_class_0 = torch.where(
        denom_class_0 > 0,
        tn / denom_class_0,
        torch.tensor(1.0 if not exclude_empty else float("nan")),
    )
    mean_recall_class_0 = torch.nanmean(recall_class_0).item()
    stddev_class_0 = np.nanstd(recall_class_0.cpu().numpy())

    if not is_mean:
        if return_std:
            return mean_recall_class_1, stddev_class_1
        else:
            return mean_recall_class_1

    # Overall mean recall (mean between the two classes)
    overall_mean_recall = (mean_recall_class_1 + mean_recall_class_0) / 2

    overall_std_recall = (stddev_class_1 + stddev_class_0) / 2

    if return_std:
        return overall_mean_recall, overall_std_recall
    else:
        return overall_mean_recall


def compute_f1_from_cumulator(
    TPs, FPs, FNs, TNs, exclude_empty=False, reduce_mean=True, is_mean=True, return_std=False
):
    """
    Compute the F1-score for both class 1 and class 0 from cumulative TPs, FPs, FNs, and TNs.
    """
    import torch

    # concatenate
    tp = torch.cat(TPs) if isinstance(TPs, (list, tuple)) else TPs
    fp = torch.cat(FPs) if isinstance(FPs, (list, tuple)) else FPs
    fn = torch.cat(FNs) if isinstance(FNs, (list, tuple)) else FNs
    tn = torch.cat(TNs) if isinstance(TNs, (list, tuple)) else TNs

    # class 1
    denom1 = 2 * tp + fp + fn
    fallback = torch.tensor(float("nan") if exclude_empty else 1.0, device=tp.device)
    f1_1 = torch.where(denom1 > 0, (2.0 * tp) / denom1, fallback)

    # class 0
    denom0 = 2 * tn + fn + fp
    f1_0 = torch.where(denom0 > 0, (2.0 * tn) / denom0, fallback)

    # per-image arrays (CPU numpy)
    f1_1_np = f1_1.cpu().numpy()
    f1_0_np = f1_0.cpu().numpy()
    if is_mean:
        per_image = (f1_1_np + f1_0_np) / 2.0
    else:
        per_image = f1_1_np

    # if user wants the raw per-image values…
    if not reduce_mean:
        return per_image

    # otherwise aggregate across images
    mean_f1 = np.nanmean(per_image)
    if return_std:
        std_f1 = np.nanstd(per_image)
        return mean_f1, std_f1
    return mean_f1


def compute_accuracy_excluding_cases(
    TPs, FPs, FNs, TNs, return_std=False, exclude_blank_case=False
):
    """
    Computes accuracy, excluding cases with a zero denominator or no ground truth positives.
    """
    import torch

    tp = torch.cat([tp for tp in TPs])
    fp = torch.cat([fp for fp in FPs])
    fn = torch.cat([fn for fn in FNs])
    tn = torch.cat([tn for tn in TNs])

    # Compute the denominator for accuracy (tp + fp + fn + tn)
    denominator = tp + fp + fn + tn

    if exclude_blank_case:
        valid_mask = (tp + fp + fn) != 0
    else:
        # Exclude cases with zero denominator or no ground truth positives
        valid_mask = ((tp + fp + fn) != 0) & ((tp + fn) != 0)

    # Compute accuracy only for valid cases
    accuracy = torch.zeros_like(denominator, dtype=torch.float)
    accuracy[valid_mask] = (tp[valid_mask] + tn[valid_mask]) / denominator[valid_mask]

    # Exclude invalid cases by setting them to NaN
    accuracy[~valid_mask] = torch.tensor(float("nan"))

    # Compute mean accuracy
    mean_accuracy = torch.nanmean(accuracy).item()

    if return_std:
        # Compute standard deviation, ignoring NaN values
        stddev_accuracy = np.nanstd(accuracy.cpu().numpy())
        return mean_accuracy, stddev_accuracy

    return mean_accuracy


def compute_precision_excluding_cases_from_cumulator(
    TPs, FPs, FNs, TNs, return_std=False, exclude_only_zero_denominator=False
):
    """
    Computes precision for cases with a non-zero denominator and excludes cases where there are no ground truth positives.
    """
    import torch

    tp = torch.cat([tp for tp in TPs])
    fp = torch.cat([fp for fp in FPs])
    fn = torch.cat([fn for fn in FNs])
    tn = torch.cat([tn for tn in TNs])

    # Compute the denominator for precision (tp + fp)
    denominator = tp + fp

    if exclude_only_zero_denominator:
        valid_mask = denominator != 0
    else:
        # Exclude cases with zero denominator or no ground truth positives
        valid_mask = (denominator != 0) & ((tp + fn) != 0)

    # Compute precision only for valid cases
    precision = torch.zeros_like(denominator, dtype=torch.float)
    precision[valid_mask] = tp[valid_mask] / denominator[valid_mask]

    # Exclude invalid cases by setting them to NaN
    precision[~valid_mask] = torch.tensor(float("nan"))

    # Compute mean precision
    mean_precision = torch.nanmean(precision).item()

    if return_std:
        # Compute standard deviation, ignoring NaN values
        stddev_precision = np.nanstd(precision.cpu().numpy())
        return mean_precision, stddev_precision

    return mean_precision


def compute_recall_excluding_cases_from_cumulator(
    TPs, FPs, FNs, TNs, return_std=False, exclude_only_zero_denominator=False
):
    """
    Computes recall for class 1, excluding cases with a zero denominator or no ground truth positives.
    """
    import torch

    tp = torch.cat([tp for tp in TPs])
    fp = torch.cat([fp for fp in FPs])
    fn = torch.cat([fn for fn in FNs])
    tn = torch.cat([tn for tn in TNs])

    # Compute the denominator for recall (tp + fn)
    denominator = tp + fn

    if exclude_only_zero_denominator:
        valid_mask = denominator != 0
    else:
        # Exclude cases with zero denominator or no ground truth positives
        valid_mask = (denominator != 0) & ((tp + fn) != 0)

    # Compute recall only for valid cases
    recall = torch.zeros_like(denominator, dtype=torch.float)
    recall[valid_mask] = tp[valid_mask] / denominator[valid_mask]

    # Exclude invalid cases by setting them to NaN
    recall[~valid_mask] = torch.tensor(float("nan"))

    # Compute mean recall
    mean_recall = torch.nanmean(recall).item()

    if return_std:
        # Compute standard deviation, ignoring NaN values
        stddev_recall = np.nanstd(recall.cpu().numpy())
        return mean_recall, stddev_recall

    return mean_recall


def compute_f1_excluding_cases_from_cumulator(
    TPs, FPs, FNs, TNs, return_std=False, exclude_only_zero_denominator=False
):
    """
    Computes F1 score for each 'case' (sample), excluding invalid cases.
    Invalid cases may be:
      - Those with zero denominator (tp + 0.5*(fp + fn) = 0).
      - Those with no ground-truth positives (tp + fn = 0), depending on the 'exclude_only_zero_denominator' flag.

    Args:
        TPs: List of tensors for true positives across batches.
        FPs: List of tensors for false positives across batches.
        FNs: List of tensors for false negatives across batches.
        TNs: List of tensors for true negatives across batches.
        return_std: Boolean indicating whether to return standard deviation.
        exclude_only_zero_denominator: If True, we only exclude cases where (2*tp + fp + fn) = 0.
                                       If False, we also exclude cases with no ground-truth positives (tp + fn = 0).

    Returns:
        mean_f1: Mean F1 across valid cases (float).
        (optional) stddev_f1: Standard deviation of F1 across valid cases (float),
                              only returned if return_std is True.
    """
    # 1. Concatenate all batches
    tp = torch.cat([tp for tp in TPs])
    fp = torch.cat([fp for fp in FPs])
    fn = torch.cat([fn for fn in FNs])
    tn = torch.cat([tn for tn in TNs])

    # 2. Compute the per-case denominator for F1 = 2*TP / (2*TP + FP + FN)
    denominator = tp + 0.5 * (fp + fn)

    # 3. Determine valid cases
    #    If 'exclude_only_zero_denominator' is True, only exclude denominator == 0.
    #    Otherwise, also exclude cases where there are no positives in ground truth (tp+fn=0).
    if exclude_only_zero_denominator:
        valid_mask = denominator != 0
    else:
        valid_mask = (denominator != 0) & ((tp + fn) != 0)

    # 4. Allocate a tensor to hold the per-case F1
    f1 = torch.zeros_like(denominator, dtype=torch.float)

    # 5. Compute F1 only for valid cases
    f1[valid_mask] = (tp[valid_mask]) / denominator[valid_mask]

    # 6. Mark invalid cases as NaN for later exclusion in mean/std computations
    f1[~valid_mask] = torch.tensor(float("nan"))

    # 7. Compute mean F1 (ignoring NaNs)
    mean_f1 = torch.nanmean(f1).item()

    if return_std:
        # 8. Compute standard deviation (ignoring NaNs)
        stddev_f1 = np.nanstd(f1.cpu().numpy())
        return mean_f1, stddev_f1


def calculate_mass_detection_imagewise_volume(y_pred, y_true):

    detection_rates = []
    for idx in range(0, y_pred.shape[-1]):
        slice_pred = y_pred[:, :, idx]
        slice_true = y_true[:, :, idx]

        structure = np.ones((3, 3), dtype=np.bool_)  # 3D connectivity
        labels_true, num_true = labell(slice_true, structure=structure)
        labels_pred, num_pred = labell(slice_pred, structure=structure)

        if num_true != 0:
            detected_masses = 0
            true_objects = find_objects(labels_true)
            pred_objects = find_objects(labels_pred)

            for i, true_slice in enumerate(true_objects):
                for j, pred_slice in enumerate(pred_objects):
                    if not check_overlap(true_slice, pred_slice):
                        continue  # Skip if bounding boxes don't overlap
                    # Calculate the IoU only for the overlapping region
                    overlap_region = tuple(
                        slice(max(t.start, p.start), min(t.stop, p.stop))
                        for t, p in zip(true_slice, pred_slice)
                    )
                    true_mass = labels_true == (i + 1)
                    pred_mass = labels_pred == (j + 1)
                    if np.any(true_mass[overlap_region]) and np.any(pred_mass[overlap_region]):
                        detected_masses += 1
                        break  # Found an overlapping mass, move to the next true mass

            detection_rate = detected_masses / num_true
            detection_rates.append(detection_rate)

    return detection_rates


def check_overlap(slice1, slice2):
    # Check if two slices overlap
    for dim1, dim2 in zip(slice1, slice2):
        if dim1.stop <= dim2.start or dim2.stop <= dim1.start:
            return False
    return True
