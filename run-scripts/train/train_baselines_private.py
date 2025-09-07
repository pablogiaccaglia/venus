import os
import pprint
from typing import Tuple

import torch
import lightning.pytorch as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from sklearn.model_selection import train_test_split

from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged
import monai

from breast_segmentation.config.settings import config
from breast_segmentation.utils.seed import set_deterministic_mode, seed_worker, reseed
from breast_segmentation.data import custom_collate_no_patches
from breast_segmentation.transforms.compose import Preprocess
from breast_segmentation.models.lightning_module import BreastSegmentationModel
from breast_segmentation.data.private_dataset import (
    PATIENT_INFO,
    get_train_val_test_dicts,
    PATIENTS_TO_EXCLUDE,
)


pp = pprint.PrettyPrinter(indent=4)


def _convert_to_l_mode(image):
    return image.convert("L")


def _section(title: str) -> None:
    print(f"\n[ {title} ]\n")


def _info(message: str) -> None:
    print(f"- {message}")


def _build_statistics_loader(batch_size: int, num_workers: int, train_dicts):
    statistics_transforms = Compose([
        LoadImaged(
            keys=["image", "label"],
            image_only=False,
            reader=monai.data.NumpyReader(),
        ),
        EnsureChannelFirstd(keys=["image", "label"]),
        Preprocess(
            keys=None,
            mode="statistics",
            dataset="private",
            subtracted_images_path_prefixes=("Dataset-arrays-4-FINAL", "Dataset-arrays-FINAL"),
            get_patches=False,
        ),
    ])

    ds = CacheDataset(data=train_dicts, transform=statistics_transforms, num_workers=num_workers)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=set_deterministic_mode(config.SEED),
        shuffle=False,
        drop_last=False,
        collate_fn=custom_collate_no_patches,
    )
    return loader


def _get_mean_std_dataloader(dataloader, masked: bool = False) -> Tuple[float, float]:
    sum_of_images = 0.0
    sum_of_squares = 0.0
    num_pixels = 0
    for batch in dataloader:
        if batch is None:
            continue
        image = batch["image"]
        if masked:
            mask = image > 0.0
            image = image[mask]
        sum_of_images += image.sum()
        sum_of_squares += (image ** 2).sum()
        num_pixels += image.numel()
    mean = (sum_of_images / num_pixels).item()
    std_dev = ((sum_of_squares / num_pixels - (sum_of_images / num_pixels) ** 2) ** 0.5).item()
    return mean, std_dev


def _build_datasets_and_loaders(
    batch_size: int,
    num_workers: int,
    mean_val: float,
    std_val: float,
    train_dicts,
    val_dicts,
    test_dicts,
):
    sub_third_images_path_prefixes = ("Dataset-arrays-4-FINAL", "Dataset-arrays-FINAL")

    test_transforms = Compose([
        LoadImaged(
            keys=["image", "label"],
            image_only=False,
            reader=monai.data.PILReader(converter=_convert_to_l_mode),
        ),
        EnsureChannelFirstd(keys=["image", "label"]),
        monai.transforms.Rotate90d(keys=["image", "label"]),
        Preprocess(
            keys=None,
            mode="test",
            dataset="private",
            subtracted_images_path_prefixes=sub_third_images_path_prefixes,
            subtrahend=mean_val,
            divisor=std_val,
            get_patches=False,
            get_boundaryloss=False,
        ),
    ])

    train_transforms = Compose([
        LoadImaged(
            keys=["image", "label"],
            image_only=False,
            reader=monai.data.PILReader(converter=_convert_to_l_mode),
        ),
        EnsureChannelFirstd(keys=["image", "label"]),
        monai.transforms.Rotate90d(keys=["image", "label"]),
        Preprocess(
            keys=None,
            mode="train",
            dataset="private",
            subtracted_images_path_prefixes=sub_third_images_path_prefixes,
            subtrahend=mean_val,
            divisor=std_val,
            get_patches=False,
            get_boundaryloss=False,
        ),
    ])

    train_ds = CacheDataset(data=train_dicts, transform=train_transforms, num_workers=num_workers)
    val_ds = CacheDataset(data=val_dicts, transform=test_transforms, num_workers=num_workers)
    test_ds = CacheDataset(data=test_dicts, transform=test_transforms, num_workers=num_workers)

    g = set_deterministic_mode(config.SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
        drop_last=False,
        collate_fn=custom_collate_no_patches,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=False,
        drop_last=False,
        collate_fn=custom_collate_no_patches,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=False,
        drop_last=False,
        collate_fn=custom_collate_no_patches,
    )

    return train_ds, train_loader, val_loader, test_loader


def train_private_baselines():
    torch.set_float32_matmul_precision("medium")
    _section("Environment & MONAI Config")
    print_config()

    batch_size = config.BATCH_SIZE
    num_workers = config.NUM_WORKERS
    checkpoints_dir = config.checkpoints_dir_private
    os.makedirs(checkpoints_dir, exist_ok=True)

    g = set_deterministic_mode(config.SEED)
    _info(f"Batch size: {batch_size}")
    _info(f"Number of workers: {num_workers}")
    _info(f"Checkpoints directory: {checkpoints_dir}")
    _info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        _info(f"Device: {torch.cuda.get_device_name()}")

    _section("Patient split & dataset dicts")
    patient_ids = list(PATIENT_INFO.keys())
    _info(f"Initial patients from PATIENT_INFO: {len(patient_ids)}")
    _info(f"Patients to exclude: {PATIENTS_TO_EXCLUDE}")
    patient_ids = [pid for pid in patient_ids if pid not in PATIENTS_TO_EXCLUDE]
    _info(f"Patients after exclusion: {len(patient_ids)}")

    SEED = config.SEED
    dataset_base_path = config.DATASET_BASE_PATH_PRIVATE
    x_train_val, x_test = train_test_split(patient_ids, test_size=0.2, random_state=SEED)
    x_train, x_val = train_test_split(x_train_val, test_size=0.25, random_state=SEED)

    _info("Creating data dictionaries with filtering logic...")
    train_dicts, val_dicts, test_dicts = get_train_val_test_dicts(dataset_base_path, x_train, x_val, x_test)
    _info(f"Train dicts: {len(train_dicts)} | Val dicts: {len(val_dicts)} | Test dicts: {len(test_dicts)}")

    _section("Preparing normalization statistics (private)")
    _info("Building statistics dataloader (NumpyReader)")
    statistics_loader = _build_statistics_loader(batch_size=batch_size, num_workers=num_workers, train_dicts=train_dicts)
    _info("Computing dataset mean/std (masked=False)")
    mean, std_dev = _get_mean_std_dataloader(statistics_loader)
    _info(f"Computed mean/std -> mean: {mean:.6f}, std: {std_dev:.6f}")

    mean_no_thorax_third_sub, std_no_thorax_third_sub = 43.14976119995117, 172.67039489746094
    _info(f"Using predefined mean/std -> mean: {mean_no_thorax_third_sub:.6f}, std: {std_no_thorax_third_sub:.6f}")

    _section("Building datasets and dataloaders")
    train_ds, train_loader, val_loader, test_loader = _build_datasets_and_loaders(
        batch_size=batch_size,
        num_workers=num_workers,
        mean_val=mean_no_thorax_third_sub,
        std_val=std_no_thorax_third_sub,
        train_dicts=train_dicts,
        val_dicts=val_dicts,
        test_dicts=test_dicts,
    )
    _info(f"Train samples: {len(train_loader.dataset)}")
    _info(f"Val samples:   {len(val_loader.dataset)}")
    _info(f"Test samples:  {len(test_loader.dataset)}")

    # FCN-FFNET
    _section("Training FCN-FFNET")
    g = reseed()
    fcn_ffnet_model = BreastSegmentationModel(
        arch="fcn_ffnet",
        encoder_name=None,
        loss_function="crossentropy2d",
        use_boundary_loss=False,
        img_size = config.IMAGE_SIZE[0],
        in_channels=config.IN_CHANNELS,
        out_classes=config.OUT_CHANNELS,
        batch_size=batch_size,
        len_train_loader=len(train_ds) // batch_size,
    )
    es = EarlyStopping(monitor="valid_loss", mode="min", patience=config.EARLY_STOPPING_PATIENCE)
    cc_fcn_ffnet_model = ModelCheckpoint(
        monitor="valid_loss",
        save_top_k=1,
        mode="min",
        filename='unet-clahe--mit2-{epoch:02d}-{valid_loss:.2f}',
        dirpath=checkpoints_dir,
        auto_insert_metric_name=False,
    )
    trainer_fcn_ffnet_model = L.Trainer(
        devices = 1,
        accelerator='auto',
        max_epochs=config.MAX_EPOCHS,
        callbacks=[es, cc_fcn_ffnet_model],
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
        gradient_clip_val=config.GRADIENT_CLIP_VAL,
        num_sanity_val_steps=1,
        deterministic=True,
    )
    trainer_fcn_ffnet_model.fit(
        fcn_ffnet_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    _info("Training complete. Loading best checkpoint and testing FCN-FFNET")
    fcn_ffnet_model = BreastSegmentationModel.load_from_checkpoint(
        cc_fcn_ffnet_model.best_model_path,
        loss_function="crossentropy2d",
        use_boundary_loss=False,
    )
    test_metrics = trainer_fcn_ffnet_model.test(
        fcn_ffnet_model,
        dataloaders=test_loader,
        verbose=False,
    )
    pp.pprint(test_metrics[0])

    # Swin-UNETR
    _section("Training Swin-UNETR")
    g = reseed()
    swin_unetr_model = BreastSegmentationModel(
        arch="swin_unetr",
        encoder_name=None,
        loss_function='soft_dice',
        use_boundary_loss=False,
        img_size = config.IMAGE_SIZE[0],
        in_channels=config.IN_CHANNELS,
        out_classes=config.OUT_CHANNELS,
        batch_size=batch_size,
        len_train_loader=len(train_ds) // batch_size,
    )
    es_swin = EarlyStopping(monitor="valid_loss", mode="min", patience=config.EARLY_STOPPING_PATIENCE)
    cc_swin_unetr_model = ModelCheckpoint(
        monitor="valid_loss",
        save_top_k=1,
        mode="min",
        filename='swin-unetr-{epoch:02d}-{valid_loss:.2f}',
        dirpath=checkpoints_dir,
        auto_insert_metric_name=False,
    )
    trainer_swin_unetr_model = L.Trainer(
        devices=1,
        accelerator='auto',
        max_epochs=config.MAX_EPOCHS,
        callbacks=[es_swin, cc_swin_unetr_model],
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
        gradient_clip_val=config.GRADIENT_CLIP_VAL,
        num_sanity_val_steps=1,
        deterministic=True,
    )
    trainer_swin_unetr_model.fit(
        swin_unetr_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    _info("Training complete. Loading best checkpoint and testing Swin-UNETR")
    swin_unetr_model = BreastSegmentationModel.load_from_checkpoint(
        cc_swin_unetr_model.best_model_path,
        loss_function='soft_dice',
        use_boundary_loss=False,
    )
    test_metrics_swin = trainer_swin_unetr_model.test(
        swin_unetr_model,
        dataloaders=test_loader,
        verbose=False,
    )
    pp.pprint(test_metrics_swin[0])

    # Unet++
    _section("Training UNet++")
    g = reseed()
    unetplusplus_model = BreastSegmentationModel(
        arch="unetplusplus",
        encoder_name=None,
        loss_function="dice_ce",
        loss_kwargs={"sigmoid": True, "lambda_dice": 0.5, "lambda_ce": 0.5},
        use_boundary_loss=False,
        img_size = config.IMAGE_SIZE[0],
        in_channels=config.IN_CHANNELS,
        out_classes=config.OUT_CHANNELS,
        batch_size=batch_size,
        len_train_loader=len(train_ds) // batch_size,
    )
    es_unetpp = EarlyStopping(monitor="valid_loss", mode="min", patience=config.EARLY_STOPPING_PATIENCE)
    cc_unetplusplus_model = ModelCheckpoint(
        monitor="valid_loss",
        save_top_k=1,
        mode="min",
        filename='unetplusplus-{epoch:02d}-{valid_loss:.2f}',
        dirpath=checkpoints_dir,
        auto_insert_metric_name=False,
    )
    trainer_unetplusplus_model = L.Trainer(
        devices=1,
        accelerator='auto',
        max_epochs=config.MAX_EPOCHS,
        callbacks=[es_unetpp, cc_unetplusplus_model],
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
        gradient_clip_val=config.GRADIENT_CLIP_VAL,
        num_sanity_val_steps=1,
        deterministic=True,
    )
    trainer_unetplusplus_model.fit(
        unetplusplus_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    _info("Training complete. Loading best checkpoint and testing UNet++")
    unetplusplus_model = BreastSegmentationModel.load_from_checkpoint(
        cc_unetplusplus_model.best_model_path,
        loss_function="dice_ce",
        use_boundary_loss=False,
    )
    test_metrics_unetpp = trainer_unetplusplus_model.test(
        unetplusplus_model,
        dataloaders=test_loader,
        verbose=False,
    )
    pp.pprint(test_metrics_unetpp[0])

    # SegNet
    _section("Training SegNet")
    g = reseed()
    segnet_model = BreastSegmentationModel(
        arch="segnet",
        encoder_name=None,
        loss_function="crossentropy2d",
        use_boundary_loss=False,
        img_size = config.IMAGE_SIZE[0],
        in_channels=config.IN_CHANNELS,
        out_classes=config.OUT_CHANNELS,
        batch_size=batch_size,
        len_train_loader=len(train_ds) // batch_size,
    )
    es_segnet = EarlyStopping(monitor="valid_loss", mode="min", patience=config.EARLY_STOPPING_PATIENCE)
    cc_segnet_model = ModelCheckpoint(
        monitor="valid_loss",
        save_top_k=1,
        mode="min",
        filename='segnet-{epoch:02d}-{valid_loss:.2f}',
        dirpath=checkpoints_dir,
        auto_insert_metric_name=False,
    )
    trainer_segnet_model = L.Trainer(
        devices=1,
        accelerator='auto',
        max_epochs=config.MAX_EPOCHS,
        callbacks=[es_segnet, cc_segnet_model],
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
        gradient_clip_val=config.GRADIENT_CLIP_VAL,
        num_sanity_val_steps=1,
        deterministic=True,
    )
    trainer_segnet_model.fit(
        segnet_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    _info("Training complete. Loading best checkpoint and testing SegNet")
    segnet_model = BreastSegmentationModel.load_from_checkpoint(
        cc_segnet_model.best_model_path,
        loss_function="crossentropy2d",
        use_boundary_loss=False,
    )
    test_metrics_segnet = trainer_segnet_model.test(
        segnet_model,
        dataloaders=test_loader,
        verbose=False,
    )
    pp.pprint(test_metrics_segnet[0])

    # Skinny
    _section("Training SkinnyNet")
    skinny_model = BreastSegmentationModel(
        arch="skinny",
        encoder_name=None,
        loss_function="dice_ce",
        loss_kwargs={"sigmoid": True, "lambda_dice": 0.5, "lambda_ce": 0.5},
        use_boundary_loss=False,
        img_size = config.IMAGE_SIZE[0],
        in_channels=config.IN_CHANNELS,
        out_classes=config.OUT_CHANNELS,
        batch_size=batch_size,
        len_train_loader=len(train_ds) // batch_size,
    )
    es_skinny = EarlyStopping(monitor="valid_loss", mode="min", patience=config.EARLY_STOPPING_PATIENCE)
    cc_skinny_model = ModelCheckpoint(
        monitor="valid_loss",
        save_top_k=1,
        mode="min",
        dirpath="./checkpoints/",
        filename="skinny_best",
        auto_insert_metric_name=False,
    )
    trainer_skinny_model = L.Trainer(
        max_epochs=config.MAX_EPOCHS,
        accelerator='auto',
        devices=1,
        callbacks=[es_skinny, cc_skinny_model],
        deterministic=True,
        precision=16,
    )
    trainer_skinny_model.fit(
        skinny_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    _info("Training complete. Loading best checkpoint and testing SkinnyNet")
    skinny_model = BreastSegmentationModel.load_from_checkpoint(
        cc_skinny_model.best_model_path,
        loss_function="dice_ce",
        use_boundary_loss=False,
    )
    test_metrics_skinny = trainer_skinny_model.test(
        skinny_model,
        dataloaders=test_loader,
        verbose=False,
    )
    pp.pprint(test_metrics_skinny[0])

    # ResNet-UNet
    _section("Training ResNet-UNet (encoder=resnet50)")
    g = reseed()
    resnet_model = BreastSegmentationModel(
        arch="UNet",
        encoder_name="resnet50",
        loss_function="dice",
        loss_kwargs={"sigmoid": True},
        use_boundary_loss=False,
        img_size = config.IMAGE_SIZE[0],
        in_channels=config.IN_CHANNELS,
        out_classes=config.OUT_CHANNELS,
        batch_size=batch_size,
        len_train_loader=len(train_ds) // batch_size,
    )
    es_resnet = EarlyStopping(monitor="valid_loss", mode="min", patience=config.EARLY_STOPPING_PATIENCE)
    cc_resnet_model = ModelCheckpoint(
        monitor="valid_loss",
        save_top_k=1,
        mode="min",
        filename='resnet-unet-{epoch:02d}-{valid_loss:.2f}',
        dirpath=checkpoints_dir,
        auto_insert_metric_name=False,
    )
    trainer_resnet_model = L.Trainer(
        devices=1,
        accelerator='auto',
        max_epochs=config.MAX_EPOCHS,
        callbacks=[es_resnet, cc_resnet_model],
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
        gradient_clip_val=config.GRADIENT_CLIP_VAL,
        num_sanity_val_steps=1,
        deterministic=True,
    )
    trainer_resnet_model.fit(
        resnet_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    _info("Training complete. Loading best checkpoint and testing ResNet-UNet")
    resnet_model = BreastSegmentationModel.load_from_checkpoint(
        cc_resnet_model.best_model_path,
        loss_function="dice",
        use_boundary_loss=False,
    )
    test_metrics_resnet = trainer_resnet_model.test(
        resnet_model,
        dataloaders=test_loader,
        verbose=False,
    )
    pp.pprint(test_metrics_resnet[0])


def main():
    train_private_baselines()


if __name__ == "__main__":
    main()


