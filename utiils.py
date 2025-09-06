from natsort import natsorted, ns
import os
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    ResizeWithPadOrCropd,
    Resized,
    MapTransform,
    ScaleIntensityd
)
import numpy as np
import wandb
import torch.nn as nn
import torch.nn.functional as F
from monai import losses
import torch
import cv2
import segmentation_models_pytorch as smp

d = {'BC1179': {'start': 58,
  'end': 111,
  'masks_len_with_mass': 54,
  'masks_len': 212,
  'images_len': 212,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.625, 0.625]},
 'AS0170': {'start': 46,
  'end': 131,
  'masks_len_with_mass': 86,
  'masks_len': 192,
  'images_len': 192,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.8203, 0.8203]},
 'ASMK0783': {'start': 81,
  'end': 116,
  'masks_len_with_mass': 36,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'D2MP9(VR)': {'start': 35,
  'end': 68,
  'masks_len_with_mass': 34,
  'masks_len': 114,
  'images_len': 114,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'D1AP7(VR)': {'start': 55,
  'end': 76,
  'masks_len_with_mass': 22,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'BR0138': {'start': 42,
  'end': 70,
  'masks_len_with_mass': 29,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7422, 0.7422]},
 'D2MP1(VR)': {'start': 47,
  'end': 61,
  'masks_len_with_mass': 15,
  'masks_len': 96,
  'images_len': 96,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'D2MP4(VR)': {'start': 57,
  'end': 128,
  'masks_len_with_mass': 72,
  'masks_len': 260,
  'images_len': 260,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'D1AP12(VR)': {'start': 22,
  'end': 35,
  'masks_len_with_mass': 14,
  'masks_len': 168,
  'images_len': 168,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7109, 0.7109]},
 'D2MP3(VR)': {'start': 132,
  'end': 144,
  'masks_len_with_mass': 13,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'BNB1172(DF)': {'start': 131,
  'end': 199,
  'masks_len_with_mass': 69,
  'masks_len': 284,
  'images_len': 284,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.6641, 0.6641]},
 'D2MP6(VR)': {'start': 24,
  'end': 50,
  'masks_len_with_mass': 27,
  'masks_len': 114,
  'images_len': 114,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'DFC1168(DF)': {'start': 141,
  'end': 172,
  'masks_len_with_mass': 32,
  'masks_len': 284,
  'images_len': 284,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.625, 0.625]},
 'DMA0247': {'start': 46,
  'end': 55,
  'masks_len_with_mass': 10,
  'masks_len': 176,
  'images_len': 176,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'HV1263(1,5)': {'start': 6,
  'end': 17,
  'masks_len_with_mass': 12,
  'masks_len': 80,
  'images_len': 80,
  'manufacturer': 'SIEMENS',
  'pixel_spacing': [0.5, 0.5]},
 'D3MP7(VR)': {'start': 36,
  'end': 71,
  'masks_len_with_mass': 36,
  'masks_len': 164,
  'images_len': 164,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.5078, 0.5078]},
 'DBA0676': {'start': 30,
  'end': 38,
  'masks_len_with_mass': 9,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'DCC0340(1,5)': {'start': 33,
  'end': 53,
  'masks_len_with_mass': 21,
  'masks_len': 96,
  'images_len': 96,
  'manufacturer': 'SIEMENS',
  'pixel_spacing': [0.5, 0.5]},
 'DTM0772(1,5)': {'start': 21,
  'end': 48,
  'masks_len_with_mass': 28,
  'masks_len': 80,
  'images_len': 80,
  'manufacturer': 'SIEMENS',
  'pixel_spacing': [0.53125, 0.53125]},
 'GA07(DF)': {'start': 61,
  'end': 82,
  'masks_len_with_mass': 22,
  'masks_len': 240,
  'images_len': 240,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7422, 0.7422]},
 'DDP0459': {'start': 96,
  'end': 126,
  'masks_len_with_mass': 31,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'GMG0961(3)': {'start': 72,
  'end': 79,
  'masks_len_with_mass': 8,
  'masks_len': 204,
  'images_len': 204,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.625, 0.625]},
 'LD0372': {'start': 61,
  'end': 130,
  'masks_len_with_mass': 70,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'PMG0761(3)': {'start': 71,
  'end': 89,
  'masks_len_with_mass': 19,
  'masks_len': 244,
  'images_len': 244,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'LA0248': {'start': 115,
  'end': 145,
  'masks_len_with_mass': 31,
  'masks_len': 284,
  'images_len': 284,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.625, 0.625]},
 'PE0468(1,5)': {'start': 19,
  'end': 27,
  'masks_len_with_mass': 9,
  'masks_len': 80,
  'images_len': 80,
  'manufacturer': 'SIEMENS',
  'pixel_spacing': [0.5, 0.5]},
 'LGM0159(1,5)': {'start': 43,
  'end': 66,
  'masks_len_with_mass': 24,
  'masks_len': 112,
  'images_len': 112,
  'manufacturer': 'SIEMENS',
  'pixel_spacing': [0.5, 0.5]},
 'MV1276(1,5)': {'start': 43,
  'end': 56,
  'masks_len_with_mass': 14,
  'masks_len': 88,
  'images_len': 88,
  'manufacturer': 'SIEMENS',
  'pixel_spacing': [0.546875, 0.546875]},
 'LAXXX': {'start': 89,
  'end': 147,
  'masks_len_with_mass': 59,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'MD0773(DF)': {'start': 118,
  'end': 151,
  'masks_len_with_mass': 34,
  'masks_len': 220,
  'images_len': 220,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.6641, 0.6641]},
 'PA0978': {'start': 44,
  'end': 58,
  'masks_len_with_mass': 15,
  'masks_len': 114,
  'images_len': 114,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'PF0473(1,5)': {'start': 47,
  'end': 57,
  'masks_len_with_mass': 11,
  'masks_len': 88,
  'images_len': 88,
  'manufacturer': 'SIEMENS',
  'pixel_spacing': [0.53125, 0.53125]},
 'VDMB0751(DF)': {'start': 75,
  'end': 156,
  'masks_len_with_mass': 82,
  'masks_len': 256,
  'images_len': 256,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.625, 0.625]},
 'SM1232': {'start': 84,
  'end': 136,
  'masks_len_with_mass': 53,
  'masks_len': 260,
  'images_len': 260,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.6641, 0.6641]},
 'ZT0279(3)': {'start': 108,
  'end': 135,
  'masks_len_with_mass': 28,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'RD0175': {'start': 82,
  'end': 121,
  'masks_len_with_mass': 40,
  'masks_len': 196,
  'images_len': 196,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.5859, 0.5859]},
 'PS0446(1,5)': {'start': 29,
  'end': 38,
  'masks_len_with_mass': 10,
  'masks_len': 88,
  'images_len': 88,
  'manufacturer': 'SIEMENS',
  'pixel_spacing': [0.5, 0.5]},
 'SM0972(DF)': {'start': 104,
  'end': 139,
  'masks_len_with_mass': 36,
  'masks_len': 240,
  'images_len': 240,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.625, 0.625]},
 'SD0462': {'start': 129,
  'end': 154,
  'masks_len_with_mass': 26,
  'masks_len': 284,
  'images_len': 284,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.625, 0.625]},
 'UFR0987': {'start': 73,
  'end': 86,
  'masks_len_with_mass': 14,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'VS0976(1,5)': {'start': 55,
  'end': 56,
  'masks_len_with_mass': 2,
  'masks_len': 88,
  'images_len': 88,
  'manufacturer': 'SIEMENS',
  'pixel_spacing': [0.5, 0.5]},
 'GMA0650': {'start': 53,
  'end': 88,
  'masks_len_with_mass': 36,
  'masks_len': 188,
  'images_len': 188,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7422, 0.7422]},
 'GLA1074': {'start': 13,
  'end': 71,
  'masks_len_with_mass': 59,
  'masks_len': 112,
  'images_len': 112,
  'manufacturer': 'SIEMENS',
  'pixel_spacing': [0.5, 0.5]},
 'CC0167': {'start': 111,
  'end': 129,
  'masks_len_with_mass': 19,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'GP0454': {'start': 70,
  'end': 87,
  'masks_len_with_mass': 18,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'LF0680': {'start': 122,
  'end': 143,
  'masks_len_with_mass': 22,
  'masks_len': 272,
  'images_len': 272,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.625, 0.625]},
 'OL1062R': {'start': 76,
  'end': 124,
  'masks_len_with_mass': 49,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'PR0760': {'start': 32,
  'end': 185,
  'masks_len_with_mass': 154,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'RM0850': {'start': 58,
  'end': 94,
  'masks_len_with_mass': 37,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7422, 0.7422]},
 'RP0178': {'start': 70,
  'end': 204,
  'masks_len_with_mass': 135,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.7031, 0.7031]},
 'TE0966': {'start': 95,
  'end': 162,
  'masks_len_with_mass': 68,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': 'GE MEDICAL SYSTEMS',
  'pixel_spacing': [0.6641, 0.6641]},
 'TI0465': {'start': 21,
  'end': 27,
  'masks_len_with_mass': 7,
  'masks_len': 88,
  'images_len': 88,
  'manufacturer': 'SIEMENS',
  'pixel_spacing': [0.5, 0.5]},
 'PF0671': {'start': 76,
  'end': 196,
  'masks_len_with_mass': 121,
  'masks_len': 208,
  'images_len': 208,
  'manufacturer': None,
  'pixel_spacing': [0.7031000256538391, 0.7031000256538391, 1.0]},
 'PRP0185': {'start': 51,
  'end': 101,
  'masks_len_with_mass': 51,
  'masks_len': 180,
  'images_len': 180,
  'manufacturer': None,
  'pixel_spacing': [0.7031000256538391, 0.7031000256538391, 1.0]},
 'GF0380': {'start': 50,
  'end': 149,
  'masks_len_with_mass': 100,
  'masks_len': 204,
  'images_len': 204,
  'manufacturer': None,
  'pixel_spacing': [0.5859000086784363, 0.5859000086784363, 1.0]},
 'SA0379': {'start': 40,
  'end': 111,
  'masks_len_with_mass': 72,
  'masks_len': 156,
  'images_len': 156,
  'manufacturer': None,
  'pixel_spacing': [0.6640999913215637,
   0.6640999913215637,
   1.000067949295044]},
 'TS0668': {'start': 73,
  'end': 105,
  'masks_len_with_mass': 33,
  'masks_len': 160,
  'images_len': 160,
  'manufacturer': None,
  'pixel_spacing': [0.7851999998092651,
   0.7851999998092651,
   0.9000015258789062]},
 'MG0477': {'start': 46,
  'end': 83,
  'masks_len_with_mass': 38,
  'masks_len': 132,
  'images_len': 132,
  'manufacturer': None,
  'pixel_spacing': [0.6640999913215637,
   0.6640999913215637,
   1.0000743865966797]},
 'PV0781': {'start': 12,
  'end': 129,
  'masks_len_with_mass': 118,
  'masks_len': 156,
  'images_len': 156,
  'manufacturer': None,
  'pixel_spacing': [0.6640999913215637,
   0.6640999913215637,
   1.000067949295044]},
 'PV1094': {'start': 57,
  'end': 122,
  'masks_len_with_mass': 66,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': None,
  'pixel_spacing': [0.7031000256538391, 0.7031000256538391, 1.0]},
 'DPA0271': {'start': 102,
  'end': 109,
  'masks_len_with_mass': 8,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': None,
  'pixel_spacing': [0.7031000256538391, 0.7031000256538391, 1.0]},
 'SAE0157': {'start': 55,
  'end': 102,
  'masks_len_with_mass': 48,
  'masks_len': 188,
  'images_len': 188,
  'manufacturer': None,
  'pixel_spacing': [0.5859000086784363, 0.5859000086784363, 1.0]},
 'TM0258': {'start': 34,
  'end': 144,
  'masks_len_with_mass': 111,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': None,
  'pixel_spacing': [0.7031000256538391, 0.7031000256538391, 1.0]},
 'VE1077': {'start': 135,
  'end': 146,
  'masks_len_with_mass': 12,
  'masks_len': 188,
  'images_len': 188,
  'manufacturer': None,
  'pixel_spacing': [0.7031000256538391,
   0.7031000256538391,
   1.0000585317611694]},
 'BV1252': {'start': 96,
  'end': 106,
  'masks_len_with_mass': 11,
  'masks_len': 188,
  'images_len': 188,
  'manufacturer': None,
  'pixel_spacing': [0.6640999913215637,
   0.6640999913215637,
   1.0000038146972656]},
 'MS0478': {'start': 65,
  'end': 168,
  'masks_len_with_mass': 104,
  'masks_len': 268,
  'images_len': 268,
  'manufacturer': None,
  'pixel_spacing': [0.7422000169754028,
   0.7422000169754028,
   0.7999954223632812]},
 'SS1281': {'start': 55,
  'end': 123,
  'masks_len_with_mass': 69,
  'masks_len': 212,
  'images_len': 212,
  'manufacturer': None,
  'pixel_spacing': [0.5468999743461609, 0.5468999743461609, 1.0]},
 'MP140270': {'start': 48,
  'end': 58,
  'masks_len_with_mass': 11,
  'masks_len': 80,
  'images_len': 80,
  'manufacturer': None,
  'pixel_spacing': [0.8035714030265808,
   0.8035714030265808,
   1.7999992370605469]},
 'SL191251': {'start': 46,
  'end': 49,
  'masks_len_with_mass': 4,
  'masks_len': 80,
  'images_len': 80,
  'manufacturer': None,
  'pixel_spacing': [0.7589285969734192,
   0.7589285969734192,
   1.8000030517578125]},
 'AM14051962': {'start': 95,
  'end': 140,
  'masks_len_with_mass': 46,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': None,
  'pixel_spacing': [0.7031000256538391, 0.7031000256538391, 1.0]},
 'MA020377': {'start': 131,
  'end': 148,
  'masks_len_with_mass': 18,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': None,
  'pixel_spacing': [0.7031000256538391, 0.7031000256538391, 1.0]},
 'PBS050277': {'start': 32,
  'end': 37,
  'masks_len_with_mass': 6,
  'masks_len': 88,
  'images_len': 88,
  'manufacturer': None,
  'pixel_spacing': [0.8088235259056091,
   0.8088235259056091,
   1.7999992370605469]},
 'SD080569': {'start': 85,
  'end': 149,
  'masks_len_with_mass': 65,
  'masks_len': 236,
  'images_len': 236,
  'manufacturer': None,
  'pixel_spacing': [0.7031000256538391, 0.7031000256538391, 1.0]},
 'CC100582': {'start': 67,
  'end': 159,
  'masks_len_with_mass': 93,
  'masks_len': 175,
  'images_len': 175,
  'manufacturer': None,
  'pixel_spacing': [0.625, 0.625, 0.9999945163726807]},
 'BP130964': {'start': 116,
  'end': 179,
  'masks_len_with_mass': 64,
  'masks_len': 196,
  'images_len': 196,
  'manufacturer': None,
  'pixel_spacing': [0.5468999743461609, 0.5468999743461609, 1.0]},
 'CF160366': {'start': 70,
  'end': 124,
  'masks_len_with_mass': 55,
  'masks_len': 212,
  'images_len': 212,
  'manufacturer': None,
  'pixel_spacing': [0.5859000086784363, 0.5859000086784363, 1.0]},
 'EA030650': {'start': 44,
  'end': 95,
  'masks_len_with_mass': 52,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': None,
  'pixel_spacing': [0.7031000256538391, 0.7031000256538391, 1.0]},
 'HF230274': {'start': 90,
  'end': 101,
  'masks_len_with_mass': 12,
  'masks_len': 204,
  'images_len': 204,
  'manufacturer': None,
  'pixel_spacing': [0.625, 0.625, 1.0]},
 'IV100377': {'start': 80,
  'end': 118,
  'masks_len_with_mass': 39,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': None,
  'pixel_spacing': [0.625, 0.625, 1.0]},
 'PV200741': {'start': 74,
  'end': 115,
  'masks_len_with_mass': 42,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': None,
  'pixel_spacing': [0.7031000256538391, 0.7031000256538391, 1.0]},
 'PA150139': {'start': 45,
  'end': 54,
  'masks_len_with_mass': 10,
  'masks_len': 96,
  'images_len': 96,
  'manufacturer': None,
  'pixel_spacing': [0.5625, 0.5625, 1.5999984741210938]},
 'RP271052': {'start': 56,
  'end': 63,
  'masks_len_with_mass': 8,
  'masks_len': 96,
  'images_len': 96,
  'manufacturer': None,
  'pixel_spacing': [0.59375, 0.59375, 2.0]},
 'SG170880': {'start': 83,
  'end': 100,
  'masks_len_with_mass': 18,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': None,
  'pixel_spacing': [0.7031000256538391, 0.7031000256538391, 1.0]},
 'AA1604': {'start': 71,
  'end': 125,
  'masks_len_with_mass': 55,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': None,
  'pixel_spacing': [0.7031000256538391, 0.7031000256538391, 1.0]},
 'BD0510': {'start': 93,
  'end': 183,
  'masks_len_with_mass': 91,
  'masks_len': 268,
  'images_len': 268,
  'manufacturer': None,
  'pixel_spacing': [0.7422000169754028, 0.7422000169754028, 1.0]},
 'CMR2703': {'start': 20,
  'end': 75,
  'masks_len_with_mass': 56,
  'masks_len': 88,
  'images_len': 88,
  'manufacturer': None,
  'pixel_spacing': [0.6597222089767456,
   0.6597222089767456,
   1.5999908447265625]},
 'CC070764': {'start': 117,
  'end': 121,
  'masks_len_with_mass': 5,
  'masks_len': 172,
  'images_len': 172,
  'manufacturer': None,
  'pixel_spacing': [0.7422000169754028,
   0.7422000169754028,
   1.000090479850769]},
 'CL1007': {'start': 39,
  'end': 89,
  'masks_len_with_mass': 51,
  'masks_len': 212,
  'images_len': 212,
  'manufacturer': None,
  'pixel_spacing': [0.5468999743461609, 0.5468999743461609, 1.0]},
 'CS300759': {'start': 29,
  'end': 98,
  'masks_len_with_mass': 70,
  'masks_len': 164,
  'images_len': 164,
  'manufacturer': None,
  'pixel_spacing': [0.6640999913215637,
   0.6640999913215637,
   0.9999688267707825]},
 'CG300765': {'start': 130,
  'end': 145,
  'masks_len_with_mass': 16,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': None,
  'pixel_spacing': [0.7031000256538391, 0.7031000256538391, 1.0]},
 'DBR270865': {'start': 87,
  'end': 173,
  'masks_len_with_mass': 87,
  'masks_len': 196,
  'images_len': 196,
  'manufacturer': None,
  'pixel_spacing': [0.8202999830245972,
   0.8202999830245972,
   1.0000746250152588]},
 'DOM130370': {'start': 66,
  'end': 96,
  'masks_len_with_mass': 31,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': None,
  'pixel_spacing': [0.7031000256538391, 0.7031000256538391, 1.0]},
 'FP211261': {'start': 99,
  'end': 122,
  'masks_len_with_mass': 24,
  'masks_len': 188,
  'images_len': 188,
  'manufacturer': None,
  'pixel_spacing': [0.6640999913215637,
   0.6640999913215637,
   1.0000680685043335]},
 'GF220280': {'start': 82,
  'end': 134,
  'masks_len_with_mass': 53,
  'masks_len': 208,
  'images_len': 208,
  'manufacturer': None,
  'pixel_spacing': [0.6640999913215637,
   0.6640999913215637,
   0.9994816184043884]},
 'HMLL020999': {'start': 82,
  'end': 161,
  'masks_len_with_mass': 80,
  'masks_len': 228,
  'images_len': 228,
  'manufacturer': None,
  'pixel_spacing': [0.625, 0.625, 1.0]},
 'IRA141057': {'start': 21,
  'end': 66,
  'masks_len_with_mass': 46,
  'masks_len': 112,
  'images_len': 112,
  'manufacturer': None,
  'pixel_spacing': [0.53125, 0.53125, 1.5999984741210938]},
 'LL010972': {'start': 74,
  'end': 81,
  'masks_len_with_mass': 8,
  'masks_len': 175,
  'images_len': 175,
  'manufacturer': None,
  'pixel_spacing': [0.5932203531265259,
   0.5932203531265259,
   1.000002145767212]},
 'MP140370': {'start': 62,
  'end': 119,
  'masks_len_with_mass': 58,
  'masks_len': 172,
  'images_len': 172,
  'manufacturer': None,
  'pixel_spacing': [0.6640999913215637,
   0.6640999913215637,
   1.000067949295044]},
 'MP160849': {'start': 25,
  'end': 174,
  'masks_len_with_mass': 150,
  'masks_len': 188,
  'images_len': 188,
  'manufacturer': None,
  'pixel_spacing': [0.6640999913215637,
   0.6640999913215637,
   1.0000649690628052]},
 'PM090847': {'start': 0,
  'end': 27,
  'masks_len_with_mass': 28,
  'masks_len': 96,
  'images_len': 96,
  'manufacturer': None,
  'pixel_spacing': [0.5625, 0.5625, 1.75]},
 'RHCL031174': {'start': 87,
  'end': 154,
  'masks_len_with_mass': 68,
  'masks_len': 188,
  'images_len': 188,
  'manufacturer': None,
  'pixel_spacing': [0.8202999830245972,
   0.8202999830245972,
   0.9999658465385437]},
 'MDSE060863': {'start': 50,
  'end': 59,
  'masks_len_with_mass': 10,
  'masks_len': 96,
  'images_len': 96,
  'manufacturer': None,
  'pixel_spacing': [0.53125, 0.53125, 1.5999984741210938]},
 'NP181051': {'start': 28,
  'end': 45,
  'masks_len_with_mass': 18,
  'masks_len': 72,
  'images_len': 72,
  'manufacturer': None,
  'pixel_spacing': [0.625, 0.625, 1.6000022888183594]}}


import numpy as np
from monai.transforms import MapTransform

class BoundingBoxSplit(MapTransform):
    def __init__(self, keys=("image", "label"), allow_missing_keys=False, bbox_size=(256, 256)):
        super().__init__(keys, allow_missing_keys)
        self.bbox_size = bbox_size

    def _positive_bounding_box(self, mask):
        """
        Computes the bounding box for a region of interest in a binary mask.
    
        Parameters:
        - mask (numpy.ndarray): A binary mask.
    
        Returns:
        - tuple: (y_min, y_max, x_min, x_max) coordinates of the bounding box.
        """
        # Find the row and column indices where the mask is 1.
        mask = mask[0]
        rows, cols = np.where(mask == 1)
        
        # If no ROI is found, return None.
        if len(rows) == 0 or len(cols) == 0:
            return None
        
        y_min, y_max = np.min(rows), np.max(rows)
        x_min, x_max = np.min(cols), np.max(cols)
        
        return y_min, y_max, x_min, x_max

    def _negative_bounding_box(self, mask):
        """
        Extracts two random bounding boxes of negative regions from a binary mask.
    
        Parameters:
        - mask (numpy.ndarray): A binary mask of shape (1, H, W).
    
        Returns:
        - list: Two tuples with (y_min, y_max, x_min, x_max) coordinates of the bounding boxes of the negative regions.
        """
        height, width = self.bbox_size[0], self.bbox_size[1]
        mask = mask[0]  # Remove the singleton dimension: (1, H, W) -> (H, W)
    
        H, W = mask.shape
    
        step_y = height // 2
        step_x = width // 2
    
        bboxes = []
        trials = 0
        max_trials = 1000  # To avoid infinite loops, though this value can be adjusted
    
        while len(bboxes) < 2:
            # Randomly sample a starting point
            y = np.random.randint(0, H - height + 1, 1)[0]
            x = np.random.randint(0, W - width + 1, 1)[0]
    
            # Align the sampled point to the nearest half-sized step grid
            y = (y // step_y) * step_y
            x = (x // step_x) * step_x
    
            window = mask[y:y+height, x:x+width]
            if np.sum(window) == 0 and (y, y+height-1, x, x+width-1) not in bboxes:
                bboxes.append((x, x+width-1, y, y+height-1))
            trials += 1
    
        return bboxes


        
    def _get_bboxes(self, mask):
        bboxes_negative = self._negative_bounding_box(mask)
        bboxes_negative = [bboxes_negative[0], bboxes_negative[1]]
        bbox_positive = self._positive_bounding_box(mask)
    
        if not bbox_positive:
            return bboxes_negative
    
        y_min, y_max, x_min, x_max = bbox_positive
        width, height = self.bbox_size
    
        y_center = (y_min + y_max) // 2
        x_center = (x_min + x_max) // 2
    
        y_min_new = max(y_center - height // 2, 0)
        y_max_new = y_min_new + height - 1
        x_min_new = max(x_center - width // 2, 0)
        x_max_new = x_min_new + width - 1
    
        # Adjust the bounding box if it extends beyond the mask's boundaries
        if y_max_new >= mask.shape[1]:
            y_max_new = mask.shape[1] - 1
            y_min_new = y_max_new - height + 1
        if x_max_new >= mask.shape[2]:
            x_max_new = mask.shape[2] - 1
            x_min_new = x_max_new - width + 1
    
        bbox_positive = [(x_min_new, x_max_new, y_min_new, y_max_new)]
        return bboxes_negative + bbox_positive

    def __call__(self, data):
        d = dict(data)

        data = []
        
            
        label = d['label']
        bboxes = self._get_bboxes(label)

        for bbox in bboxes:
            xmin, xmax, ymin, ymax = bbox
            new_d= {}
            # Crop using bounding box
            # Update slicing to handle the singleton dimension: (1, H, W)
            new_d['image'] = d["image"][:, ymin:ymax+1, xmin:xmax+1]
            new_d['label'] = label[:, ymin:ymax+1, xmin:xmax+1]
                    
            # Adjust meta-data for cropped image and label
            new_d["image_meta_dict"] = dict(d["image_meta_dict"])
            new_d["image_meta_dict"]["original_affine"] = d["image_meta_dict"]["affine"]
            affine_adjust = np.array([[1, 0, 0, xmin], [0, 1, 0, ymin], [0, 0, 1, 0], [0, 0, 0, 1]])
            new_d["image_meta_dict"]["affine"] = d["image_meta_dict"]["affine"] @ affine_adjust
    
            new_d["label_meta_dict"] = dict(d["label_meta_dict"])
            new_d["label_meta_dict"]["original_affine"] = d["label_meta_dict"]["affine"]
            new_d["label_meta_dict"]["affine"] = d["label_meta_dict"]["affine"] @affine_adjust

            data.append(new_d)
        return data


def get_filenames(suffix, base_path, patient_ids, remove_black_samples=False):
    filenames = []
    for patient_id in patient_ids:
        path = os.path.join(base_path, patient_id) + "/" + suffix + "/"
        files = [os.path.join(path, p) for p in natsorted(os.listdir(path), alg=ns.IGNORECASE)]

        if remove_black_samples:
            files = filter_samples_sample_aware(files, patient_id)
        filenames += files   
        
    return filenames

def filter_samples_sample_aware(files, patient_id):
    start, end = d[patient_id]['start'], d[patient_id]['end']
    return files[start+1:end]
    
def filter_samples(images_filenames, masks_filenames):
    filtered_images_filenames = []
    filtered_masks_filenames = []

    for ima_path, mask_path in zip(images_filenames, masks_filenames):
          mask = np.load(mask_path)

          if mask.sum() != 0:
              filtered_images_filenames.append(ima_path)
              filtered_masks_filenames.append(mask_path)


    return filtered_images_filenames, filtered_masks_filenames

class PreprocessForBackbone(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    def __init__(self, encoder_name, **kwargs):
        super(PreprocessForBackbone, self).__init__(**kwargs)
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, 'imagenet')

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            #d[key] = d[key].repeat(3, 1, 1)
            d[key] = np.transpose(
                    self.preprocessing_fn(
                            np.transpose(
                                    d[key].repeat(3, 1, 1)
                                    , (1, 2, 0)))
                    , (2, 0, 1)).astype(np.float32)
        return d

class SetType(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __init__(self, **kwargs):
        super( SetType, self).__init__(**kwargs)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            #d[key] = d[key].repeat(3, 1, 1)
            d[key] = d[key].astype(np.uint8)
        return d

class DiceLoss(nn.Module):
    def __init__(self, weight = None, size_average = True):
        super(DiceLoss, self).__init__()
        self.helper_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits = True)

    def forward(self, inputs, targets, smooth = 1):
        # flatten label and prediction tensors
        targets.requires_grad = True
        inputs_norm = (F.logsigmoid(inputs).exp()>0.5).float()

        inputs_norm = inputs_norm.contiguous().view(inputs_norm.shape[0], -1)
        targets = targets.contiguous().view(targets.shape[0], -1)

        mask = (targets != 0)
        nonvoid = mask.sum(dim = 1)

        # if there are batches with only void pixels, the gradients should be 0 for them
        dice = torch.zeros(inputs_norm.shape[0], device = inputs_norm.device)
        nonvoid_batches = nonvoid.nonzero(as_tuple = True)[0]

        if len(nonvoid_batches) > 0:
            for b in nonvoid_batches:
                inputs_nonvoid = inputs_norm[b][mask[b]]
                targets_nonvoid = targets[b][mask[b]]

                intersection = (inputs_nonvoid * targets_nonvoid).sum()
                dice[b] = (2. * intersection + smooth) / (inputs_nonvoid.sum() + targets_nonvoid.sum() + smooth)

        else:
            return targets.sum() * 0
            #return self.helper_loss(inputs, targets)

        return (1 - dice).mean()


def compute_iou(y_true, y_pred, class_id, reduction='micro'):
    """
    Compute Intersection over Union for a specific class

    Args:
    y_true (torch.Tensor): batch of ground truth, 4D tensor (first dimension is batch size)
    y_pred (torch.Tensor): batch of prediction, 4D tensor (first dimension is batch size)
    class_id (int): the class to compute IoU for
    reduction (str): the method of reduction across the batch, can be 'micro' or 'micro image-wise'

    Returns:
    torch.Tensor: IoU score
    """

    def compute_iou_single(y_true_single, y_pred_single, class_id_single):
        y_true_class = torch.where(y_true_single == class_id_single, 1, 0)
        y_pred_class = torch.where(y_pred_single == class_id_single, 1, 0)

        intersection = torch.logical_and(y_true_class, y_pred_class)
        union = torch.logical_or(y_true_class, y_pred_class)

        union_sum = torch.sum(union)
        if union_sum == 0:
            # Both prediction and ground truth are empty
            iou_score = torch.tensor([1.0]) 
        else:
            iou_score = torch.sum(intersection).float() / union_sum.float()

        return iou_score

    assert reduction in ['micro', 'micro_image_wise'], "Reduction method should be either 'micro' or 'micro_image_wise'"

    if reduction == 'micro':
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        return torch.tensor(compute_iou_single(y_true, y_pred, class_id)).float()

    elif reduction == 'micro_image_wise':
        iou_scores = torch.tensor([compute_iou_single(y, p, class_id) for y, p in zip(y_true, y_pred)], dtype=torch.float32)
        return torch.nanmean(iou_scores)  # Using nanmean to ignore NaN values


def class_specific_accuracy_score(preds, targets, class_id = 1, eps = 1e-7, reduction = 'mean',
                                  averaging = 'micro'):
    """
    Compute the class-specific accuracy score.

    Parameters
    ----------
    preds : torch.Tensor
        The predicted values from the model. Assumes values are [0, 1].
    targets : torch.Tensor
        The ground truth values. Assumes values are in {0, 1}.
    class_of_interest : int, optional
        The class for which to compute the accuracy. Assumes values are in {0, 1}. Default is 1.
    eps : float, optional
        Small value to prevent division by zero. Default is 1e-7.
    reduction : str, optional
        Specifies the reduction to apply to the output: 'mean', 'sum' or 'none'.
        'none': no reduction will be applied.
        'mean': the sum of the output will be divided by the number of elements in the output.
        'sum': the output will be summed. Default is 'mean'.
    averaging : str, optional
        Specifies the type of averaging to use: 'micro' or 'imagewise'.
        'micro': Calculate metrics globally by counting the total true positives, false negatives, and false positives.
        'imagewise': Calculate metrics for each instance, and find their average.

    Returns
    -------
    score : torch.Tensor
        The computed class-specific accuracy score.
    """

    if class_id == 1:
        # Compute True Positive (TP), False Positive (FP) and False Negative (FN)
        TP = (preds * targets).sum(dim = (1, 2, 3))
        FP = (preds * (1 - targets)).sum(dim = (1, 2, 3))
        FN = ((1 - preds) * targets).sum(dim = (1, 2, 3))
    else:
        # Compute True Negative (TN), False Positive (FP) and False Negative (FN)
        TN = ((1 - preds) * (1 - targets)).sum(dim = (1, 2, 3))
        FP = ((1 - preds) * targets).sum(dim = (1, 2, 3))
        FN = (preds * (1 - targets)).sum(dim = (1, 2, 3))

    if averaging == 'micro':
        if class_id == 1:
            # Compute micro-average accuracy
            score = TP.sum() / (TP.sum() + FP.sum() + FN.sum() + eps)
        else:
            # Compute micro-average accuracy
            score = TN.sum() / (TN.sum() + FP.sum() + FN.sum() + eps)
    elif averaging == 'micro_image_wise':
        if class_id == 1:
            # Compute image-wise micro-average accuracy
            score = TP / (TP + FP + FN + eps)
        else:
            # Compute image-wise micro-average accuracy
            score = TN / (TN + FP + FN + eps)

    if reduction == 'mean':
        score = score.mean()
    elif reduction == 'sum':
        score = score.sum()


    return score