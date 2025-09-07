"""Composition of transforms for different pipeline stages."""

import copy
import torch
import numpy as np
import monai
from monai.transforms import (
    MapTransform,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
    ResizeWithPadOrCropd,
    HistogramNormalized,
    MedianSmooth,
    Rotate90d,
    DeleteItemsd,
    ScaleIntensityd,
)
from monai.data import MetaTensor, PILReader
from typing import Dict, Any, List, Optional, Tuple, Union


def pil_grayscale_converter(image):
    """Convert PIL image to grayscale - replaces lambda for pickling compatibility."""
    return image.convert("L")


from ..metrics.boundaryloss.dataloader import dist_map_transform
from .preprocessing import (
    RemoveThorax,
    RemoveBottom,
    TrimSides,
    FilterBySize,
    ThresholdBlack,
    MedianSmooth as MedianSmoothTransform,
    AdaptiveCropBreasts,
    BoundingBoxSplit,
)
from ..config.settings import config


class FilterByDim(MapTransform):
    """Filter samples based on image dimensions -"""

    def __init__(self, keys):
        super().__init__(keys)

    def filter_by_dim(self, data):
        # Implement your filtering condition here
        # For example, filter out images with a certain property:

        # OLD VALUE WAS 180
        keep_sample = data["processed_label"].shape[1] > 100
        return keep_sample

    def __call__(self, data):
        # Apply the filter function to determine if the sample should be kept
        keep_sample = torch.tensor([self.filter_by_dim(data)])
        data["keep_sample"] = torch.cat((data["keep_sample"], keep_sample), dim=0)
        return data


class FilterByMean(MapTransform):
    """Filter samples based on mean intensity value -"""

    def __init__(self, keys, mean_threshold, start_pos):
        super().__init__(keys)
        self.mean_threshold = mean_threshold
        self.start_pos = start_pos

    def filter_by_mean(self, data):
        # print(data['processed_image'].std())
        keep_sample = data["processed_image"][:, self.start_pos :, :].std() > self.mean_threshold
        return keep_sample

    def __call__(self, data):
        # Apply the filter function to determine if the sample should be kept
        keep_sample = torch.tensor([self.filter_by_mean(data)])
        if data["keep_sample"] and not keep_sample:
            data["keep_sample"] = keep_sample

        return data


class Resize(MapTransform):
    """Custom Resize transform."""

    def __init__(self, step: str, keys: List[str] = ["image", "label"], spatial_size=256, **kwargs):
        super().__init__(keys, **kwargs)
        self.spatial_size = spatial_size
        self.resize = monai.transforms.Resize(spatial_size=spatial_size, mode="nearest-exact")
        self.step = step

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)

        image_key = "processed_image" if self.step == "preliminary" else "image"
        label_key = "processed_label" if self.step == "preliminary" else "label"

        image_shape = d[image_key].shape
        dim_before_resize = torch.tensor(image_shape)

        dim_before_resize_dict_key = (
            "dim_before_resize_preliminary"
            if self.step == "preliminary"
            else "dim_before_resize_final"
        )
        spatial_size_info_dict_key = (
            "spatial_size_info_preliminary"
            if self.step == "preliminary"
            else "spatial_size_info_final"
        )

        original_spatial_dim = torch.tensor([image_shape[1], image_shape[2]], dtype=torch.int16)

        d[image_key] = self.resize(d[image_key])
        if self.step != "preliminary":
            d["processed_image"] = self.resize(d["processed_image"])
        if d["has_mask"]:
            d[label_key] = self.resize(d[label_key])
            if self.step != "preliminary":
                d["processed_label"] = self.resize(d["processed_label"])

        d[spatial_size_info_dict_key] = torch.cat(
            (d[spatial_size_info_dict_key], original_spatial_dim), dim=0
        )
        d[dim_before_resize_dict_key] = torch.cat(
            (d[dim_before_resize_dict_key], dim_before_resize), dim=0
        )

        if self.step == "preliminary":
            d["preliminary_target_size"] = torch.cat(
                (d["preliminary_target_size"], torch.tensor(self.spatial_size)), dim=0
            )
        return d


class CropToSquare(MapTransform):
    """Crop image to square shape -"""

    def __init__(self, keys, shrink_factor, black_threshold):
        super().__init__(keys)
        self.shrink_factor = shrink_factor
        self.black_threshold = black_threshold  # Pixel intensity threshold for 'almost black'

    def __call__(self, data):
        d = dict(data)
        image = d["processed_image"]
        label = d["processed_label"]

        d["dim_before_crop"] = torch.cat((d["dim_before_crop"], torch.tensor(image.shape)), dim=0)

        # Crop the image along the longest dimension to remove almost black regions
        max_intensity = np.max(image[0], axis=0)  # Max intensity for each column

        valid_columns = np.argwhere(max_intensity > self.black_threshold).flatten()

        if len(valid_columns) == 0:
            x_min = 0
            x_max = image.shape[2]
        else:
            x_min, x_max = valid_columns[0], valid_columns[-1]

            if x_min - 30 >= 0:
                x_min = x_min - 30
            if x_max + 30 < image.shape[2]:
                x_max = x_max + 30

        y_min = self.shrink_factor
        y_max = image.shape[1]

        # Crop both image and label
        image = image[:, y_min:y_max, x_min:x_max]
        label = label[:, y_min:y_max, x_min:x_max]

        # Determine the new longest and shortest dimensions
        new_longest_dim = max(image.shape[1], image.shape[2])
        new_shortest_dim = min(image.shape[1], image.shape[2])

        # Pad to make a square
        pad_bottom = new_longest_dim - image.shape[1] if image.shape[1] < new_longest_dim else 0
        pad_right = new_longest_dim - image.shape[2] if image.shape[2] < new_longest_dim else 0

        d["processed_image"] = np.pad(image, ((0, 0), (0, pad_bottom), (0, pad_right)), "constant")
        d["processed_label"] = np.pad(label, ((0, 0), (0, pad_bottom), (0, pad_right)), "constant")

        # Save the crop coordinates and dimensions
        crop_coords = torch.tensor([y_min, y_max, x_min, x_max], dtype=torch.int16)
        pad_post_crop_coords = torch.tensor(
            [[0, 0], [0, pad_bottom], [0, pad_right]], dtype=torch.int16
        )
        d["crop_coords"] = torch.cat((d["crop_coords"], crop_coords), dim=0)
        d["pad_post_crop_coords"] = torch.cat(
            (d["pad_post_crop_coords"], pad_post_crop_coords), dim=0
        )

        return d


class PrepareSample(MapTransform):
    """Prepare sample for training/inference."""

    def __init__(self, target_size, subtracted_images, patches, **kwargs):
        super().__init__(**kwargs)
        self.resize = Resized(
            keys=["image", "label"], spatial_size=target_size, mode="nearest-exact"
        )
        self.patches = patches
        self.resize_original = Resized(
            keys=["original_image", "original_label"],
            spatial_size=target_size,
            mode="nearest-exact",
        )
        self.subtracted_images = subtracted_images
        self.loadimage = monai.transforms.LoadImage(
            ensure_channel_first=True,
            reader=monai.data.PILReader(converter=pil_grayscale_converter),
        )

    def prepare_with_patches(self, data):
        """Prepare data with patches - includes trim_breast_coords step."""
        trim_breast_coords = data["trim_breast_coords"].tolist()
        thorax_crop_coords = data["thorax_crop_coords"].tolist()
        bottom_crop_coords = data["bottom_crop_coords"].tolist()
        crop_coords = data["crop_coords"].tolist()
        pad_post_crop_coords = data["pad_post_crop_coords"].tolist()

        if self.subtracted_images:
            image_path = data["image_meta_dict"]["subtracted_filename_or_obj"]
            if image_path.endswith(".npy"):
                image = np.load(image_path)
                image = np.expand_dims(image, 0)
                image = MetaTensor(image)
            else:
                label_path = data["label_meta_dict"]["subtracted_filename_or_obj"]
                image = self.loadimage(image_path)
                image = monai.transforms.Rotate90()(image)
                image = MetaTensor(image)

                label = self.loadimage(label_path)
                label = monai.transforms.Rotate90()(label)
                label = MetaTensor(label)
                data["label"] = label

            data["image"] = image

        data = self.resize(data)
        data = self.resize_original(data)

        image = data["image"]
        original_image = data["original_image"]

        target_size = data["preliminary_target_size"].tolist()

        x1, y1, x2, y2 = thorax_crop_coords
        image = image[:, y1:, :]
        original_image = original_image[:, y1:, :]

        start, end = trim_breast_coords
        image = image[:, :, start:end]
        original_image = original_image[:, :, start:end]

        x1, y1, x2, y2 = bottom_crop_coords
        image = image[:, :y2, :]
        original_image = original_image[:, :y2, :]

        y_min, y_max, x_min, x_max = crop_coords
        image = image[:, y_min:y_max, x_min:x_max]
        original_image = original_image[:, y_min:y_max, x_min:x_max]

        before1, before2, before3 = (
            pad_post_crop_coords[0],
            pad_post_crop_coords[1],
            pad_post_crop_coords[2],
        )
        image = np.pad(
            image,
            ((before1[0], before1[1]), (before2[0], before2[1]), (before3[0], before3[1])),
            "constant",
        )
        original_image = np.pad(
            original_image,
            ((before1[0], before1[1]), (before2[0], before2[1]), (before3[0], before3[1])),
            "constant",
        )

        data["image"] = image
        data["processed_image"] = original_image

        if data["has_mask"]:
            label = data["label"]
            original_label = data["original_label"]

            x1, y1, x2, y2 = thorax_crop_coords
            label = label[:, y1:, :]
            original_label = original_label[:, y1:, :]

            start, end = trim_breast_coords
            label = label[:, :, start:end]
            original_label = original_label[:, :, start:end]

            x1, y1, x2, y2 = bottom_crop_coords
            label = label[:, :y2, :]
            original_label = original_label[:, :y2, :]

            y_min, y_max, x_min, x_max = crop_coords
            label = label[:, y_min:y_max, x_min:x_max]
            original_label = original_label[:, y_min:y_max, x_min:x_max]

            before1, before2, before3 = (
                pad_post_crop_coords[0],
                pad_post_crop_coords[1],
                pad_post_crop_coords[2],
            )
            label = np.pad(
                label,
                ((before1[0], before1[1]), (before2[0], before2[1]), (before3[0], before3[1])),
                "constant",
            )
            original_label = np.pad(
                original_label,
                ((before1[0], before1[1]), (before2[0], before2[1]), (before3[0], before3[1])),
                "constant",
            )

            data["label"] = label
            data["processed_label"] = original_label

        return data

    def prepare_without_patches(self, data):
        """Prepare data without patches - EXCLUDES trim_breast_coords step."""
        trim_breast_coords = data["trim_breast_coords"].tolist()
        thorax_crop_coords = data["thorax_crop_coords"].tolist()
        bottom_crop_coords = data["bottom_crop_coords"].tolist()
        crop_coords = data["crop_coords"].tolist()
        pad_post_crop_coords = data["pad_post_crop_coords"].tolist()

        if self.subtracted_images:
            image_path = data["image_meta_dict"]["subtracted_filename_or_obj"]
            if image_path.endswith(".npy"):
                image = np.load(image_path)
                image = np.expand_dims(image, 0)
                image = MetaTensor(image)
                data["image"] = image
            else:  # FOR BREADM
                label_path = data["label_meta_dict"]["subtracted_filename_or_obj"]
                image = self.loadimage(image_path)
                image = monai.transforms.Rotate90()(image)
                image = MetaTensor(image)

                label = self.loadimage(label_path)
                label = monai.transforms.Rotate90()(label)
                label = MetaTensor(label)
                data["label"] = label
                data["image"] = image

        data = self.resize(data)
        data = self.resize_original(data)

        image = data["image"]
        original_image = data["original_image"]

        target_size = data["preliminary_target_size"].tolist()

        x1, y1, x2, y2 = thorax_crop_coords
        image = image[:, y1:, :]
        original_image = original_image[:, y1:, :]

        # start, end = trim_breast_coords
        # image = image[:,:, start:end]

        x1, y1, x2, y2 = bottom_crop_coords
        image = image[:, :y2, :]
        original_image = original_image[:, :y2, :]

        y_min, y_max, x_min, x_max = crop_coords
        image = image[:, y_min:y_max, x_min:x_max]
        original_image = original_image[:, y_min:y_max, x_min:x_max]

        before1, before2, before3 = (
            pad_post_crop_coords[0],
            pad_post_crop_coords[1],
            pad_post_crop_coords[2],
        )
        image = np.pad(
            image,
            ((before1[0], before1[1]), (before2[0], before2[1]), (before3[0], before3[1])),
            "constant",
        )
        original_image = np.pad(
            original_image,
            ((before1[0], before1[1]), (before2[0], before2[1]), (before3[0], before3[1])),
            "constant",
        )

        data["image"] = image
        data["processed_image"] = original_image

        if data["has_mask"]:
            label = data["label"]
            original_label = data["original_label"]

            x1, y1, x2, y2 = thorax_crop_coords
            label = label[:, y1:, :]
            original_label = original_label[:, y1:, :]

            # start, end = trim_breast_coords
            # label = label[:,:, start:end]

            x1, y1, x2, y2 = bottom_crop_coords
            label = label[:, :y2, :]
            original_label = original_label[:, :y2, :]

            y_min, y_max, x_min, x_max = crop_coords
            label = label[:, y_min:y_max, x_min:x_max]
            original_label = original_label[:, y_min:y_max, x_min:x_max]

            before1, before2, before3 = (
                pad_post_crop_coords[0],
                pad_post_crop_coords[1],
                pad_post_crop_coords[2],
            )
            label = np.pad(
                label,
                ((before1[0], before1[1]), (before2[0], before2[1]), (before3[0], before3[1])),
                "constant",
            )
            original_label = np.pad(
                original_label,
                ((before1[0], before1[1]), (before2[0], before2[1]), (before3[0], before3[1])),
                "constant",
            )

            data["label"] = label
            data["processed_label"] = original_label

        return data

    def __call__(self, data):
        if self.patches:
            return self.prepare_with_patches(data)
        else:
            return self.prepare_without_patches(data)


class Preprocess(MapTransform):
    """Main preprocessing pipeline that orchestrates all transforms."""

    def __init__(
        self,
        subtrahend=48.85898971557617,
        divisor=123.9007568359375,
        median_smooth_radius=2,
        hist_norm_num_bins=None,
        rm_thorax_threshold=None,
        rm_thorax_margin=None,
        rm_bottom_threshold=None,
        rm_bottom_margin=None,
        threshold_black_threshold=1300,
        threshold_black_value=0,
        trim_sides_threshold=20000,
        trim_sides_tolerance=20,
        bbox_size=(384, 384),
        pad_spatial_size=[512, 512],
        target_size_preliminary=(512, 512),
        target_size_final=(256, 256),
        mode="train",
        subtracted_images_path_prefixes=None,
        has_mask=True,
        dataset="BREADM",
        get_patches=False,
        get_boundaryloss=False,
        **kwargs,
    ):

        super().__init__(**kwargs)

        from ..config.settings import config

        self.subtracted_images_path_prefixes = subtracted_images_path_prefixes
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.get_patches = get_patches
        self.dataset = dataset

        # Set dataset-specific parameters
        if dataset == "BREADM":
            self.subtracted_images = True
            hist_norm_num_bins = hist_norm_num_bins or config.HIST_NORM_NUM_BINS
            rm_thorax_threshold = rm_thorax_threshold or config.RM_THORAX_THRESHOLD
            rm_thorax_margin = rm_thorax_margin or config.RM_THORAX_MARGIN
            rm_bottom_threshold = rm_bottom_threshold or config.RM_BOTTOM_THRESHOLD
            rm_bottom_margin = rm_bottom_margin or config.RM_BOTTOM_MARGIN
        elif dataset == "private":
            if subtracted_images_path_prefixes:
                self.subtracted_images = True
            else:
                self.subtracted_images = False
            hist_norm_num_bins = hist_norm_num_bins or config.PRIVATE_HIST_NORM_NUM_BINS
            rm_thorax_threshold = rm_thorax_threshold or config.PRIVATE_RM_THORAX_THRESHOLD
            rm_thorax_margin = rm_thorax_margin or config.PRIVATE_RM_THORAX_MARGIN
            rm_bottom_threshold = rm_bottom_threshold or config.PRIVATE_RM_BOTTOM_THRESHOLD
            rm_bottom_margin = rm_bottom_margin or config.PRIVATE_RM_BOTTOM_MARGIN
        else:
            # Default behavior for other datasets
            if subtracted_images_path_prefixes:
                self.subtracted_images = True
            else:
                self.subtracted_images = False
            hist_norm_num_bins = hist_norm_num_bins or config.HIST_NORM_NUM_BINS
            rm_thorax_threshold = rm_thorax_threshold or config.RM_THORAX_THRESHOLD
            rm_thorax_margin = rm_thorax_margin or config.RM_THORAX_MARGIN
            rm_bottom_threshold = rm_bottom_threshold or config.RM_BOTTOM_THRESHOLD
            rm_bottom_margin = rm_bottom_margin or config.RM_BOTTOM_MARGIN

        # Set mean_threshold and start_pos based on patches
        if get_patches:
            mean_threshold = 50
            start_pos = 75
        else:
            mean_threshold = 35
            start_pos = 40

        # Initialize all transforms
        """Initialize all individual transforms."""
        self.median_smooth = MedianSmoothTransform(
            keys=["processed_image"], radius=median_smooth_radius
        )
        self.histogram_normalized = HistogramNormalized(
            keys=["processed_image"], num_bins=hist_norm_num_bins
        )
        self.remove_thorax = RemoveThorax(
            keys=["processed_image", "processed_label"],
            threshold=rm_thorax_threshold,
            margin=rm_thorax_margin,
            value=0,
        )
        self.remove_bottom = RemoveBottom(
            keys=["processed_image", "processed_label"],
            threshold=rm_bottom_threshold,
            margin=rm_bottom_margin,
            value=0,
        )
        self.threshold_black = ThresholdBlack(
            keys=["processed_image"],
            threshold=threshold_black_threshold,
            value=threshold_black_value,
        )
        self.trim_sides = TrimSides(
            keys=["processed_image", "processed_label"],
            threshold=trim_sides_threshold,
            tolerance=trim_sides_tolerance,
        )
        self.normalize = monai.transforms.NormalizeIntensityd(
            keys=["image"], subtrahend=self.subtrahend, divisor=self.divisor
        )
        self.bbox_split = BoundingBoxSplit(keys=["image", "label"], bbox_size=bbox_size)
        self.pad = monai.transforms.SpatialPadd(
            keys=["image", "label"], spatial_size=pad_spatial_size
        )
        self.convert3d = monai.transforms.RepeatChanneld(keys=["image"], repeats=3)
        self.prepare_image = PrepareSample(
            keys=None,
            target_size=target_size_preliminary,
            subtracted_images=self.subtracted_images,
            patches=self.get_patches,
        )
        self.adaptiveCropBreasts = AdaptiveCropBreasts(keys=["processed_image", "processed_label"])
        self.crop = CropToSquare(
            keys=["processed_image", "processed_label"], shrink_factor=15, black_threshold=400
        )
        self.resizePreliminary = Resize(
            step="preliminary", keys=["image", "label"], spatial_size=target_size_preliminary
        )
        self.resizePreliminaryCleanImg = Resized(
            keys=["image", "label"], spatial_size=target_size_preliminary, mode="nearest-exact"
        )
        self.resizeFinal = Resize(
            step="final", keys=["image", "label"], spatial_size=target_size_final
        )
        self.foreground_mask = monai.transforms.ForegroundMaskd(keys=["image"], invert=True)
        self.filterbyDim = FilterByDim(keys=["processed_image", "processed_label"])
        self.filterbyMean = FilterByMean(
            keys=["image", "label"], mean_threshold=mean_threshold, start_pos=start_pos
        )
        self.has_mask = has_mask
        self.get_boundaryloss = get_boundaryloss

        # Normalization with subtrahend and divisor
        if self.subtrahend is not None and self.divisor is not None:
            self.normalize = monai.transforms.NormalizeIntensityd(
                keys=["image"], subtrahend=self.subtrahend, divisor=self.divisor
            )
        else:
            self.normalize = ScaleIntensityd(keys=["image"])

        # Compose transforms based on mode
        self.mode = mode
        self._compose_transforms()

        # Distance transform for boundary loss
        self.disttransform = dist_map_transform([1, 1], 2)

    def _compose_transforms(self):
        """Compose transforms based on mode and settings."""
        # Define all transform pipelines

        # Train functions with patches
        self.train_functions_patches = [
            self.resizePreliminary,
            self.median_smooth,
            self.histogram_normalized,
            self.resizePreliminaryCleanImg,
            self.remove_thorax,
            self.adaptiveCropBreasts,
            self.remove_bottom,
            self.filterbyDim,
            self.crop,
            self.prepare_image,
            self.resizeFinal,
            self.filterbyMean,
            self.normalize,
        ]

        # Train functions without patches
        self.train_functions_no_patches = [
            self.resizePreliminary,
            self.median_smooth,
            self.histogram_normalized,
            self.resizePreliminaryCleanImg,
            self.remove_thorax,
            self.remove_bottom,
            self.filterbyDim,
            self.crop,
            self.prepare_image,
            self.resizeFinal,
            self.filterbyMean,
            self.normalize,
        ]

        # Test functions with patches
        self.test_functions_patches = [
            self.resizePreliminary,
            self.median_smooth,
            self.histogram_normalized,
            self.resizePreliminaryCleanImg,
            self.remove_thorax,
            self.adaptiveCropBreasts,
            self.remove_bottom,
            self.filterbyDim,
            self.crop,
            self.prepare_image,
            self.resizeFinal,
            self.filterbyMean,
            self.normalize,
        ]

        # Test functions without patches
        self.test_functions_no_patches = [
            self.resizePreliminary,
            self.median_smooth,
            self.histogram_normalized,
            self.resizePreliminaryCleanImg,
            self.remove_thorax,
            self.remove_bottom,
            self.filterbyDim,
            self.crop,
            self.prepare_image,
            self.resizeFinal,
            self.filterbyMean,
            self.normalize,
        ]

        # Statistics functions with patches
        self.statistics_functions_patches = [
            self.resizePreliminary,
            self.median_smooth,
            self.histogram_normalized,
            self.resizePreliminaryCleanImg,
            self.remove_thorax,
            self.adaptiveCropBreasts,
            self.remove_bottom,
            self.filterbyDim,
            self.crop,
            self.prepare_image,
            self.resizeFinal,
            self.filterbyMean,
        ]

        # Statistics functions without patches
        self.statistics_functions_no_patches = [
            self.resizePreliminary,
            self.median_smooth,
            self.histogram_normalized,
            self.resizePreliminaryCleanImg,
            self.remove_thorax,
            self.remove_bottom,
            self.filterbyDim,
            self.crop,
            self.prepare_image,
            self.resizeFinal,
            self.filterbyMean,
        ]

        # Select the appropriate transform pipeline based on mode and patches
        if self.mode == "train":
            if self.get_patches:
                self.transforms = Compose(self.train_functions_patches)
            else:
                self.transforms = Compose(self.train_functions_no_patches)
        elif self.mode == "test":
            if self.get_patches:
                self.transforms = Compose(self.test_functions_patches)
            else:
                self.transforms = Compose(self.test_functions_no_patches)
        elif self.mode == "statistics":
            if self.get_patches:
                self.transforms = Compose(self.statistics_functions_patches)
            else:
                self.transforms = Compose(self.statistics_functions_no_patches)

    def __call__(self, data: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Apply preprocessing pipeline."""
        # Initialize tracking tensors
        data["thorax_crop_coords"] = torch.tensor([], dtype=torch.int16)
        data["dim_before_thorax_crop"] = torch.tensor([], dtype=torch.int16)

        data["trim_breast_coords"] = torch.tensor([], dtype=torch.int16)
        data["dim_before_breast_crop"] = torch.tensor([], dtype=torch.int16)

        data["bottom_crop_coords"] = torch.tensor([], dtype=torch.int16)
        data["dim_before_bottom_crop"] = torch.tensor([], dtype=torch.int16)

        data["pad_post_crop_coords"] = torch.tensor([], dtype=torch.int16)

        data["crop_coords"] = torch.tensor([], dtype=torch.int16)
        data["dim_before_crop"] = torch.tensor([], dtype=torch.int16)
        data["preliminary_target_size"] = torch.tensor([], dtype=torch.int16)

        data["spatial_size_info_preliminary"] = torch.tensor([], dtype=torch.int16)
        data["spatial_size_info_final"] = torch.tensor([], dtype=torch.int16)
        data["dim_before_resize_final"] = torch.tensor([], dtype=torch.int16)
        data["dim_before_resize_preliminary"] = torch.tensor([], dtype=torch.int16)

        data["processed_image"] = copy.deepcopy(data["image"])
        data["original_image"] = copy.deepcopy(data["image"])

        data["keep_sample"] = torch.tensor([], dtype=torch.bool)
        data["has_mask"] = MetaTensor(self.has_mask)

        if self.has_mask:
            data["processed_label"] = copy.deepcopy(data["label"])
            data["original_label"] = copy.deepcopy(data["label"])

        else:
            data["processed_label"] = np.zeros_like(data["image"])
            data["original_label"] = np.zeros_like(data["image"])
            data["label"] = np.zeros_like(data["image"])

            data["processed_label"] = MetaTensor(data["processed_label"])
            data["original_label"] = MetaTensor(data["original_label"])
            data["label"] = MetaTensor(data["label"])

        if self.subtracted_images_path_prefixes:
            pfx1, pfx2 = (
                self.subtracted_images_path_prefixes[0],
                self.subtracted_images_path_prefixes[1],
            )
            data["image_meta_dict"]["subtracted_filename_or_obj"] = data["image_meta_dict"][
                "filename_or_obj"
            ].replace(pfx1, pfx2)
            if self.has_mask:
                data["label_meta_dict"]["subtracted_filename_or_obj"] = data["label_meta_dict"][
                    "filename_or_obj"
                ].replace(pfx1, pfx2)

        # Apply transforms
        data = self.transforms(data)

        # Post-processing
        if self.get_patches:

            if self.dataset == "BREADM":
                if data[0]["label"].max() > 1.1:
                    data[0]["label"] = data[0]["label"] / 255.0

                if data[1]["label"].max() > 1.1:
                    data[1]["label"] = data[1]["label"] / 255.0

            c, h, w = data[0]["image"].shape

            data[0]["image_meta_dict"]["spatial_shape"] = np.array([h, w])
            data[0]["label_meta_dict"]["spatial_shape"] = np.array([h, w])

            data[1]["image_meta_dict"]["spatial_shape"] = np.array([h, w])
            data[1]["label_meta_dict"]["spatial_shape"] = np.array([h, w])

            data[0]["image_meta_dict"]["original_channel_dim"] = MetaTensor(
                data[0]["image_meta_dict"]["original_channel_dim"]
            )
            data[0]["label_meta_dict"]["original_channel_dim"] = MetaTensor(
                data[0]["label_meta_dict"]["original_channel_dim"]
            )

            data[1]["image_meta_dict"]["original_channel_dim"] = MetaTensor(
                data[1]["image_meta_dict"]["original_channel_dim"]
            )
            data[1]["label_meta_dict"]["original_channel_dim"] = MetaTensor(
                data[1]["label_meta_dict"]["original_channel_dim"]
            )

            del data[0]["original_image"]
            del data[1]["original_image"]

            if self.has_mask:
                if self.get_boundaryloss:
                    boundary = self.disttransform(data[0]["label"][0])
                    data[0]["boundary"] = boundary

                    boundary = self.disttransform(data[1]["label"][0])
                    data[1]["boundary"] = boundary
                del data[0]["original_label"]
                del data[1]["original_label"]

                data[0]["has_mass"] = MetaTensor(np.sum(data[0]["label"]) != 0)
                data[1]["has_mass"] = MetaTensor(np.sum(data[1]["label"]) != 0)

        else:

            if self.dataset == "BREADM":
                if data["label"].max() > 1.1:
                    data["label"] = data["label"] / 255.0

            c, h, w = data["image"].shape

            data["image_meta_dict"]["spatial_shape"] = np.array([h, w])
            data["label_meta_dict"]["spatial_shape"] = np.array([h, w])

            data["image_meta_dict"]["original_channel_dim"] = MetaTensor(
                data["image_meta_dict"]["original_channel_dim"]
            )
            data["label_meta_dict"]["original_channel_dim"] = MetaTensor(
                data["label_meta_dict"]["original_channel_dim"]
            )

            if self.has_mask:
                if self.get_boundaryloss:
                    boundary = self.disttransform(data["label"][0])
                    data["boundary"] = boundary
                data["has_mass"] = MetaTensor(np.sum(data["label"]) != 0)
                del data["original_label"]

            del data["original_image"]

        return data
