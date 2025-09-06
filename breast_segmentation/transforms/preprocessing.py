"""Preprocessing transforms for breast segmentation."""

import numpy as np
import torch
import monai
from monai.transforms import MapTransform
from typing import Dict, Any, Tuple, Optional, List
import cv2
from skimage import filters
import copy
import numpy as np
import torch
import monai
from monai.transforms import (
    MapTransform, Compose, LoadImaged, EnsureChannelFirstd, 
    ScaleIntensityRanged, Resized, ResizeWithPadOrCropd,
    HistogramNormalized, MedianSmooth, LabelToContourd,
    ForegroundMaskd, Rotate90d
)
from monai.data import MetaTensor
from typing import Dict, Any, List, Tuple, Optional, Union
import cv2
from PIL import Image

from boundaryloss.dataloader import dist_map_transform


class RemoveThorax(MapTransform):
    """Remove thorax region from breast images - """

    def __init__(self, threshold, value, margin=0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.value = value
        self.margin = margin

    def remove_upper_portion_get_coords(self, image):
        # IMAGE IS C, H, W
        # Step 1: Find the vertical middle line of the image
        middle_x = image.shape[2] // 2

        non_zero_y = 0
        for y in reversed(range(image.shape[1])):
            if image[:,y, middle_x] > 0:
                non_zero_y = y
                break

        return non_zero_y

    def __call__(self, data):
        d = dict(data)
        image_to_threshold = d['processed_image']
        dim_before_thorax_crop = torch.tensor(image_to_threshold.shape)
        image_to_threshold = np.where(image_to_threshold < self.threshold, self.value, image_to_threshold)
        y_coord = self.remove_upper_portion_get_coords(image_to_threshold)-self.margin
        thorax_crop_coords = torch.tensor([0, y_coord, image_to_threshold.shape[2], image_to_threshold.shape[1]], dtype=torch.int16)  # (x1, y1, x2, y2)
        image = d['processed_image']
        image = image[:,y_coord:, :]

        if data['has_mask']:
            mask = d['processed_label']
            mask = mask[:,y_coord:, :]
            d['processed_label'] = mask

        d['processed_image'] = image
        d['thorax_crop_coords'] = torch.cat((d['thorax_crop_coords'], thorax_crop_coords), dim=0)
        d['dim_before_thorax_crop'] = torch.cat((d['dim_before_thorax_crop'], dim_before_thorax_crop), dim=0)
        return d


class RemoveBottom(MapTransform):
    """Remove bottom portion of the image - """

    def __init__(self, threshold, value, margin, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.value = value
        self.margin = margin

    def remove_lower_portion_get_coords(self, image):
        # Step 2: Starting from the bottom, find the first non-zero pixel
        non_zero_y = image.shape[1]-1
        for y in reversed(range(image.shape[1])):  # Start from the bottom
            if np.sum(image[:,y,:]) > 0:
                non_zero_y = y
                break

        return non_zero_y

    def __call__(self, data):
        d = dict(data)
        image_to_threshold = d['processed_image']

        image_to_threshold = np.where(image_to_threshold < self.threshold, self.value, image_to_threshold)
        y_coord = self.remove_lower_portion_get_coords(image_to_threshold)+self.margin

        bottom_crop_coords = torch.tensor([0, 0, image_to_threshold.shape[2], y_coord], dtype=torch.int16)  # (x1, y1, x2, y2)
        image = d['processed_image']
        dim_before_bottom_crop = torch.tensor(image.shape)
        image = image[:,:y_coord, :]

        if d['has_mask']:
            mask = d['processed_label']
            mask = mask[:,:y_coord, :]
            d['processed_label'] = mask

        d['processed_image'] = image
        d['bottom_crop_coords'] = torch.cat((d['bottom_crop_coords'], bottom_crop_coords), dim=0)
        d['dim_before_bottom_crop'] = torch.cat((d['dim_before_bottom_crop'], dim_before_bottom_crop), dim=0)

        return d


class TrimSides(MapTransform):
    """Trim black regions from the sides of the image."""
    
    def __init__(self, keys: Tuple[str, ...], threshold: float, tolerance: int, **kwargs):
        super().__init__(keys, **kwargs)
        self.threshold = threshold
        self.tolerance = tolerance

    def trim_sides(self, image_data: np.ndarray, threshold: float, tolerance: int) -> Tuple[int, int]:
        """Calculate trim coordinates."""
        # Calculate the sum of pixel values across the channel axis for each column
        col_sum = np.sum(image_data, axis=0).sum(axis=0)
        
        # Find indices where the sum exceeds the threshold
        x_start = np.argmax(col_sum > threshold)
        x_end = len(col_sum) - np.argmax(col_sum[::-1] > threshold) - 1
        
        # Apply tolerance
        x_start = max(0, x_start - tolerance)
        x_end = min(len(col_sum) - 1, x_end + tolerance)
        
        return x_start, x_end

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        
        # Initialize tracking tensors if not present
        if 'trim_coords' not in d:
            d['trim_coords'] = torch.empty((0,), dtype=torch.int16)
        
        image = d.get('processed_image', d['image'])
        x_start, x_end = self.trim_sides(image, self.threshold, self.tolerance)
        
        # Crop the image
        cropped_image = image[:, :, x_start:x_end+1]
        d['processed_image'] = cropped_image
        
        # Crop mask if present
        if d.get('has_mask', False) and 'processed_label' in d:
            mask = d['processed_label']
            cropped_mask = mask[:, :, x_start:x_end+1]
            d['processed_label'] = cropped_mask
        
        # Store coordinates
        trim_coords = torch.tensor([x_start, x_end+1], dtype=torch.int16)
        d['trim_coords'] = torch.cat((d['trim_coords'], trim_coords), dim=0)
        
        return d


class FilterBySize(MapTransform):
    """Filter out images with extreme aspect ratios."""
    
    def __init__(self, max_ratio: float, keys: Optional[List[str]] = None, **kwargs):
        super().__init__(keys, **kwargs)
        self.max_ratio = max_ratio
        self.delete = monai.transforms.DeleteItemsd(keys=['image', 'label'])

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        c, h, w = d['image'].shape

        if h >= w:
            if w == 0 or self.max_ratio < h/w:
                return self.delete(d)
        elif w >= h:
            if h == 0 or self.max_ratio < w/h:
                return self.delete(d)

        return d


class MedianSmooth(MapTransform):
    """Apply median smoothing to images."""
    
    def __init__(self, radius, **kwargs):
        super().__init__(**kwargs)
        self.median_smooth = monai.transforms.MedianSmooth(radius=radius)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        d['processed_image'] = self.median_smooth(d['processed_image'])
        return d


class ThresholdBlack(MapTransform):
    """Threshold black regions in the image."""
    
    def __init__(self, threshold: float, value: float, keys: Optional[List[str]] = None, **kwargs):
        super().__init__(keys, **kwargs)
        self.threshold = threshold
        self.value = value

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        
        image = d.get('processed_image', d['image'])
        d['processed_image'] = monai.data.MetaTensor(
            np.where(image < self.threshold, self.value, image)
        )
        
        if d.get('has_mask', False) and 'processed_label' in d:
            label = d['processed_label']
            d['processed_label'] = monai.data.MetaTensor(
                np.where(label < self.threshold, self.value, label)
            )
        
        return d

class BoundingBoxSplit(MapTransform):
    """Split images into patches based on bounding boxes."""
    
    def __init__(self, keys: Tuple[str, ...] = ("image", "label"), 
                 allow_missing_keys: bool = False, 
                 bbox_size: Tuple[int, int] = (256, 256)):
        super().__init__(keys, allow_missing_keys)
        self.bbox_size = bbox_size

    def pad_image(self, image: torch.Tensor) -> torch.Tensor:
        """Pad image to ensure it can contain bounding boxes."""
        _, h, w = image.shape
        width, height = self.bbox_size
        
        # Calculate padding needed
        pad_height = max(height - h, 0)
        pad_width = max(width - w, 0)
        
        if pad_height > 0 or pad_width > 0:
            padding = [(0, 0), (0, pad_height), (0, pad_width)]
            padded_image = np.pad(image, padding, mode='constant', constant_values=0)
            return torch.tensor(padded_image)
        
        return image

    def _positive_bounding_box(self, mask: torch.Tensor) -> Optional[Tuple[int, int, int, int]]:
        """Compute bounding box for positive region in mask."""
        mask = mask[0]  # Remove channel dimension
        rows, cols = np.where(mask == 1)
        
        if len(rows) == 0 or len(cols) == 0:
            return None
        
        y_min, y_max = np.min(rows), np.max(rows)
        x_min, x_max = np.min(cols), np.max(cols)
        
        return y_min, y_max, x_min, x_max

    def _negative_bounding_box(self, mask: torch.Tensor, num_boxes: int = 2) -> List[Tuple[int, int, int, int]]:
        """Extract random bounding boxes from negative regions."""
        height, width = self.bbox_size
        mask = mask[0]  # Remove channel dimension
        H, W = mask.shape
        
        step_y = height // 2
        step_x = width // 2
        
        bboxes = []
        trials = 0
        max_trials = 50
        
        while len(bboxes) < num_boxes and trials < max_trials:
            y = np.random.randint(0, H - height + 1, 1)[0]
            x = np.random.randint(0, W - width + 1, 1)[0]
            
            # Align to grid
            y = (y // step_y) * step_y
            x = (x // step_x) * step_x
            
            window = mask[y:y+height, x:x+width]
            if np.sum(window) == 0 and (x, x+width-1, y, y+height-1) not in bboxes:
                bboxes.append((x, x+width-1, y, y+height-1))
            trials += 1
        
        return bboxes

    def _get_bboxes(self, mask: torch.Tensor) -> List[Tuple[int, int, int, int]]:
        """Get all bounding boxes for the mask."""
        if mask.sum() == 0:
            return self._negative_bounding_box(mask, num_boxes=1)
        
        bbox_negative = self._negative_bounding_box(mask, num_boxes=1)
        bbox_positive = self._positive_bounding_box(mask)
        
        if not bbox_positive:
            return bbox_negative
        
        # Adjust positive bbox to desired size
        y_min, y_max, x_min, x_max = bbox_positive
        width, height = self.bbox_size
        
        pos_width = x_max - x_min + 1
        pos_height = y_max - y_min + 1
        
        # Center the bbox around the positive region
        x_min_new = max(x_min - (width - pos_width) // 2, 0)
        y_min_new = max(y_min - (height - pos_height) // 2, 0)
        
        x_max_new = x_min_new + width - 1
        y_max_new = y_min_new + height - 1
        
        # Adjust if extends beyond boundaries
        if y_max_new >= mask.shape[1]:
            y_max_new = mask.shape[1] - 1
            y_min_new = max(y_max_new - height + 1, 0)
        if x_max_new >= mask.shape[2]:
            x_max_new = mask.shape[2] - 1
            x_min_new = max(x_max_new - width + 1, 0)
        
        # Ensure positive region is included
        x_min_new = min(x_min_new, x_min)
        y_min_new = min(y_min_new, y_min)
        x_max_new = max(x_max_new, x_max)
        y_max_new = max(y_max_new, y_max)
        
        bbox_positive = [(x_min_new, x_max_new, y_min_new, y_max_new)]
        return bbox_negative + bbox_positive

    def __call__(self, data: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        d = dict(data)
        
        # Pad images
        d['image'] = self.pad_image(d['image'])
        d['label'] = self.pad_image(d['label'])
        
        label = d['label']
        bboxes = self._get_bboxes(label)
        
        if len(bboxes) == 0:
            return d
        
        data_list = []
        for bbox in bboxes:
            xmin, xmax, ymin, ymax = bbox
            new_d = d.copy()
            
            # Crop using bounding box
            new_d['image'] = torch.tensor(d["image"][:, ymin:ymax+1, xmin:xmax+1])
            new_d['label'] = torch.tensor(label[:, ymin:ymax+1, xmin:xmax+1])
            
            # Adjust metadata
            if "image_meta_dict" in d:
                new_d["image_meta_dict"] = dict(d["image_meta_dict"])
                new_d["image_meta_dict"]["original_affine"] = d["image_meta_dict"]["affine"]
                
                affine_adjust = np.array([[1, 0, 0, xmin], [0, 1, 0, ymin], [0, 0, 1, 0], [0, 0, 0, 1]])
                new_d["image_meta_dict"]["affine"] = d["image_meta_dict"]["affine"] @ affine_adjust
                new_d["image_meta_dict"]["affine"] = MetaTensor(new_d["image_meta_dict"]['affine'])
            
            if "label_meta_dict" in d:
                new_d["label_meta_dict"] = dict(d["label_meta_dict"])
                new_d["label_meta_dict"]["original_affine"] = d["label_meta_dict"]["affine"]
                
                new_d["label_meta_dict"]["affine"] = d["label_meta_dict"]["affine"] @ affine_adjust
                new_d["label_meta_dict"]["affine"] = MetaTensor(new_d["label_meta_dict"]['affine'])
            
            data_list.append(new_d)
        
        return data_list


class AdaptiveCropBreasts(MapTransform):
    """Adaptively crop breast regions from images - exact implementation from original AdaptiveCropBreasts2."""
    
    def __init__(self, keys: List[str] = ['processed_image','processed_label'], 
                 strict_boundary_perc: float = 0.001):
        super().__init__(keys)
        self.strict_boundary_perc = strict_boundary_perc

    def find_strict_breast_region(self, half_image_sum: np.ndarray, 
                                peak_index: int, 
                                total_width: int) -> Tuple[int, int]:
        """Find strict boundaries of breast region."""
        # Set a stricter percentage of the peak value to consider as the breast boundary
        peak_value = half_image_sum[peak_index]
        boundary_threshold = peak_value * self.strict_boundary_perc

        # Find the left boundary of the breast region
        left_boundary = peak_index
        while left_boundary > 0 and half_image_sum[left_boundary] > boundary_threshold:
            left_boundary -= 1

        # Find the right boundary of the breast region
        right_boundary = peak_index
        while right_boundary < total_width and half_image_sum[right_boundary] > boundary_threshold:
            right_boundary += 1

        return left_boundary, right_boundary

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        d = dict(data)
        data_list = []

        x1, y1, x2, y2 = d['thorax_crop_coords']

        image = copy.deepcopy(d['image'])
        mask = copy.deepcopy(d['label'])

        processed_image = copy.deepcopy(d['processed_image'])
        processed_mask = copy.deepcopy(d['processed_label'])

        dim_before_breast_crop = torch.tensor(image.shape)
        image = image[:,y1:, :]
        mask = mask[:,y1:, :]

        image_for_check = image[:, 20:, :]

        # Calculate the vertical sum for the left and right halves
        mid_point = image_for_check.shape[2] // 2

        left_half_sum = image_for_check.sum(axis=(0, 1))[:mid_point]
        right_half_sum = image_for_check.sum(axis=(0, 1))[mid_point:]

        # Find the peak in each half
        left_peak_index = np.argmax(left_half_sum)
        right_peak_index = np.argmax(right_half_sum) + mid_point

        # Find the breast regions
        left_breast_boundaries = self.find_strict_breast_region(left_half_sum, left_peak_index, mid_point)
        right_breast_boundaries = self.find_strict_breast_region(right_half_sum, right_peak_index - mid_point, image.shape[2] - mid_point)

        # Extract the breast regions
        left_breast_region_image_strict = processed_image[:, :, left_breast_boundaries[0]:left_breast_boundaries[1]]
        left_breast_region_mask_strict = processed_mask[:, :, left_breast_boundaries[0]:left_breast_boundaries[1]]

        right_breast_region_mask_strict = processed_mask[:, :, right_breast_boundaries[0] + mid_point:right_breast_boundaries[1] + mid_point]
        right_breast_region_image_strict = processed_image[:, :, right_breast_boundaries[0] + mid_point:right_breast_boundaries[1] + mid_point]

        left_breast_trim_coords = torch.tensor([left_breast_boundaries[0], left_breast_boundaries[1]], dtype=torch.int16)
        right_breast_trim_coords = torch.tensor([right_breast_boundaries[0] + mid_point, right_breast_boundaries[1] + mid_point], dtype=torch.int16)

        regions = [
            (left_breast_region_image_strict, left_breast_region_mask_strict, left_breast_trim_coords),
            (right_breast_region_image_strict, right_breast_region_mask_strict, right_breast_trim_coords)
        ]

        # Loop through the two largest regions to crop the image and mask
        for i, region in enumerate(regions):
            new_d = d.copy()

            new_d['processed_image'] = region[0]
            new_d['processed_label'] = region[1]

            # Adjust meta-data for cropped image and label if meta-data is available
            if "image_meta_dict" in d and "label_meta_dict" in d:
                left_boundary = left_breast_boundaries[0] if i == 0 else right_breast_boundaries[0] + mid_point
                affine_adjust = np.array([[1, 0, 0, -left_boundary], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

                new_d["image_meta_dict"] = dict(d["image_meta_dict"])
                new_d["image_meta_dict"]["original_affine"] = d["image_meta_dict"]["affine"]
                new_d["image_meta_dict"]["original_affine"] = MetaTensor(new_d["image_meta_dict"]["original_affine"])

                new_d["image_meta_dict"]["affine"] = d["image_meta_dict"]["affine"] @ affine_adjust
                new_d["image_meta_dict"]["affine"] = MetaTensor(new_d["image_meta_dict"]["affine"])

                new_d["label_meta_dict"] = dict(d["label_meta_dict"])
                new_d["label_meta_dict"]["original_affine"] = d["label_meta_dict"]["affine"]
                new_d["label_meta_dict"]["original_affine"] = MetaTensor(new_d["label_meta_dict"]["original_affine"])

                new_d["label_meta_dict"]["affine"] = d["label_meta_dict"]["affine"] @ affine_adjust
                new_d["label_meta_dict"]["affine"] = MetaTensor(new_d["label_meta_dict"]["affine"])

                new_d['trim_breast_coords'] = torch.cat((new_d['trim_breast_coords'], region[2]), dim=0)
                new_d['dim_before_breast_crop'] = torch.cat((d['dim_before_breast_crop'], dim_before_breast_crop), dim=0)
            data_list.append(new_d)

        return data_list
