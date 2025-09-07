import os
import sys
import pprint
import numpy as np

import torch
from sklearn.model_selection import train_test_split

import monai
from monai.config import print_config
from monai.data import CacheDataset
from monai.transforms import LoadImaged, EnsureChannelFirstd, Compose

from breast_segmentation.config.settings import config
from breast_segmentation.utils.seed import set_deterministic_mode
from breast_segmentation.transforms.compose import Preprocess
from breast_segmentation.inference.private_dataset_aware_test import (
    test_dataset_aware_ensemble,
    test_dataset_aware_no_patches,
    test_dataset_aware_fusion,
)
from breast_segmentation.metrics.losses import (
    CABFL, SurfaceLossBinary, AsymmetricUnifiedFocalLoss,
    AsymmetricFocalLoss, AsymmetricFocalTverskyLoss, SoftDiceLoss,
)


pp = pprint.PrettyPrinter(indent=4)


def _section(title: str) -> None:
    print(f"\n[ {title} ]\n")


def _info(message: str) -> None:
    print(f"- {message}")


def main():
    torch.set_float32_matmul_precision('medium')
    _section("Environment & MONAI Config")
    print_config()

    # Register loss classes for checkpoint loading
    sys.modules['__main__'].CABFL = CABFL
    sys.modules['__main__'].SurfaceLossBinary = SurfaceLossBinary
    sys.modules['__main__'].AsymmetricUnifiedFocalLoss = AsymmetricUnifiedFocalLoss
    sys.modules['__main__'].AsymmetricFocalLoss = AsymmetricFocalLoss
    sys.modules['__main__'].AsymmetricFocalTverskyLoss = AsymmetricFocalTverskyLoss
    sys.modules['__main__'].SoftDiceLoss = SoftDiceLoss

    _section("Data Setup - Private Dataset")
    NUM_WORKERS = config.NUM_WORKERS
    SEED = config.SEED
    USE_SUBTRACTED = True

    dataset_base_path = config.DATASET_BASE_PATH_PRIVATE
    CHECKPOINTS_DIR = config.checkpoints_dir_private
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    _info(f"Checkpoint directory verified: {CHECKPOINTS_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _info(f"Using device: {device}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    print("Using private dataset backend functions")
    patient_ids = os.listdir(dataset_base_path)
    print(f"Initial patients from PATIENT_INFO: {len(patient_ids)}")
    print(f"Patients after exclusion: {len(patient_ids)}")

    x_train_val, x_test = train_test_split(patient_ids, test_size=0.2, random_state=SEED)
    x_train, x_val = train_test_split(x_train_val, test_size=0.25, random_state=SEED)
    print(f"Dataset base path: {dataset_base_path}")
    print(f"Total patients: {len(patient_ids)}")
    print(f"Train patients: {len(x_train)}")
    print(f"Validation patients: {len(x_val)}")
    print(f"Test patients: {len(x_test)}")
    print(f"Test patient IDs: {x_test}")

    _section("Create Test Datasets")
    sub_third_images_path_prefixes = (config.DATASET_BASE_PATH_PRIVATE, "Dataset-arrays-FINAL")

    mean_no_thorax_third_sub = 43.1498
    std_no_thorax_third_sub = 172.6704
    mean_patches_sub = 86.13536834716797
    std_patches_sub = 238.13461303710938

    test_transforms_no_thorax_third_sub = Compose([
        LoadImaged(keys=["image", "label"], image_only=False, reader=monai.data.NumpyReader()),
        EnsureChannelFirstd(keys=["image", "label"]),
        Preprocess(
            keys=None,
            mode='test',
            dataset="private",
            subtracted_images_path_prefixes=sub_third_images_path_prefixes,
            subtrahend=mean_no_thorax_third_sub,
            divisor=std_no_thorax_third_sub,
            get_patches=False,
            get_boundaryloss=True,
        ),
    ])

    test_transforms_patches_sub = Compose([
        LoadImaged(keys=["image", "label"], image_only=False, reader=monai.data.NumpyReader()),
        EnsureChannelFirstd(keys=["image", "label"]),
        Preprocess(
            keys=None,
            mode='test',
            dataset="private",
            subtracted_images_path_prefixes=sub_third_images_path_prefixes,
            subtrahend=mean_patches_sub,
            divisor=std_patches_sub,
            get_patches=True,
            get_boundaryloss=True,
        ),
    ])
    print("Test transforms created")

    datasets = {}
    for patient_id in x_test:
        print(patient_id)
        pid_list = [patient_id]

        from breast_segmentation.data.private_dataset import get_filenames
        images_fnames, _ = get_filenames(
            suffix="images",
            base_path=config.DATASET_BASE_PATH_PRIVATE,
            patient_ids=pid_list,
            remove_black_samples=False,
            get_random_samples_and_remove_black_samples=False,
            random_samples_indexes_list=None,
        )
        labels_fnames, _ = get_filenames(
            suffix="masks",
            base_path=config.DATASET_BASE_PATH_PRIVATE,
            patient_ids=pid_list,
            remove_black_samples=False,
            get_random_samples_and_remove_black_samples=False,
            random_samples_indexes_list=None,
            remove_picked_samples=False,
        )

        test_dicts = [{"image": i, "label": l} for i, l in zip(images_fnames, labels_fnames)]
        no_thorax_sub_test_ds = CacheDataset(data=test_dicts, transform=test_transforms_no_thorax_third_sub, num_workers=NUM_WORKERS)
        patches_sub_test_ds = CacheDataset(data=test_dicts, transform=test_transforms_patches_sub, num_workers=NUM_WORKERS)

        datasets[pid_list[0]] = {
            "no_thorax_sub_test_ds": no_thorax_sub_test_ds,
            "patches_sub_test_ds": patches_sub_test_ds,
        }

    _section("Model Checkpoint Paths")
    model_paths = {
        'venus_large': f'{CHECKPOINTS_DIR}/venus-large-best.ckpt',
        'unetplusplus': f'{CHECKPOINTS_DIR}/unetplusplus_model.ckpt',
        'skinny': f'{CHECKPOINTS_DIR}/skinny_model.ckpt',
        'resnet50': f'{CHECKPOINTS_DIR}/resnet50-model.ckpt',
        'fcn': f'{CHECKPOINTS_DIR}/fcn_ffnet_model.ckpt',
        'segnet': f'{CHECKPOINTS_DIR}/segnet_model_large.ckpt',
        'swin': f'{CHECKPOINTS_DIR}/swin_model.ckpt',
        'resnet18-patches': f'{CHECKPOINTS_DIR}/resnet18-patches-model.ckpt',
    }

    print("Available models:")
    for name, path in model_paths.items():
        if os.path.exists(path):
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: {path} (not found)")

    _section("Test Baseline Models")
    baseline_tests = [
        ('unetplusplus', 'UNet++', 'unetplusplus'),
        ('skinny', 'SkinnyNet', 'skinny'),
        ('fcn', 'FCN', 'unetplusplus'),
        ('segnet', 'SegNet', 'segnet'),
        ('swin', 'Swin-UNETR', 'swin_unetr'),
        ('resnet50', 'ResNet50', 'resnet50'),
    ]

    baseline_results = {}
    for model_key, model_name, arch_name in baseline_tests:
        if os.path.exists(model_paths[model_key]):
            print(f"Testing {model_name} model...")
            result = test_dataset_aware_no_patches(
                model_path=model_paths[model_key],
                patient_ids=x_test,
                datasets=datasets,
                dataset_key="no_thorax_sub_test_ds",
                filter=False,
                get_scores_for_statistics=False,
                get_only_masses=False,
                arch_name=arch_name,
                strict=True,
                subtracted=True,
            )
            baseline_results[model_key] = result
            print(f"\n{model_name} Results:")
            pp.pprint(result)
        else:
            print(f"{model_paths[model_key]} not found.")
            baseline_results[model_key] = None
    print(f"\nCompleted {len([r for r in baseline_results.values() if r is not None])} baseline model tests.")

    _section("Test VENUS Model")
    if os.path.exists(os.path.join(CHECKPOINTS_DIR, "PRIVATE-FUSION-SUB-CABL-NO-DECODER-ATTENTION-FINAL.ckpt")):
        print("Testing VENUS Large ...")
        scores_for_statistics_fusion_large2 = test_dataset_aware_fusion(
            model_path=os.path.join(CHECKPOINTS_DIR, "PRIVATE-FUSION-SUB-CABL-NO-FUSION-FINAL.ckpt"),
            patient_ids=x_test,
            datasets=datasets,
            whole_dataset_key="no_thorax_sub_test_ds",
            patches_dataset_key="patches_sub_test_ds",
            use_simple_fusion=True,
            use_decoder_attention=True,
            strict=True,
            filter=False,
            subtracted=True,
            get_scores_for_statistics=False,
            get_only_masses=False,
            base_channels=64,
        )
        print("\nVENUS Large:")
        pp.pprint(scores_for_statistics_fusion_large2)
    else:
        print("venus-large.ckpt not found.")
        scores_for_statistics_fusion_large2 = None

    _section("Test Ensemble Model")
    ensemble_tests = [
        ('venus_large', 'resnet18-patches', False, 64, 'VENUS Large + ResNet18 patches'),
        ('venus_large', 'resnet18-patches', True, 64, 'VENUS Large + ResNet18 patches (filtered)'),
    ]

    ensemble_results = {}
    for whole_key, patches_key, use_filter, base_channels, description in ensemble_tests:
        if os.path.exists(model_paths[whole_key]) and os.path.exists(model_paths[patches_key]):
            print(f"Testing Ensemble: {description}...")
            result = test_dataset_aware_ensemble(
                model_whole_path=model_paths[whole_key],
                model_patches_path=model_paths[patches_key],
                patient_ids=x_test,
                datasets=datasets,
                whole_dataset_key="no_thorax_sub_test_ds",
                patches_dataset_key="patches_sub_test_ds",
                filter=use_filter,
                get_scores_for_statistics=False,
                get_only_masses=False,
                subtracted=True,
                base_channels=base_channels,
                use_decoder_attention = True,
                use_simple_fusion = True
            )
            ensemble_key = f"{whole_key}+{patches_key}{'_filtered' if use_filter else ''}"
            ensemble_results[ensemble_key] = result
            print(f"\n{description} Results:")
            pp.pprint(result)
        else:
            print(f"Required models not found for ensemble: {description}")
            ensemble_key = f"{whole_key}+{patches_key}{'_filtered' if use_filter else ''}"
            ensemble_results[ensemble_key] = None
    print(f"\nCompleted {len([r for r in ensemble_results.values() if r is not None])} ensemble model tests.")


if __name__ == "__main__":
    main()


