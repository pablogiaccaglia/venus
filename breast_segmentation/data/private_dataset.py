"""
Private dataset data loading utilities - exact 1:1 implementation from reference notebook.

This module contains all the custom filtering logic, patient information, and data 
loading functions specifically for the private dataset, following the exact approach
from the reference notebook training_monai-private-DECEMBER2024-baselines (1).ipynb
"""

import os
import random
from typing import List, Tuple, Dict, Optional, Any
from natsort import natsorted, ns


# Patient dictionary with start/end information
PATIENT_INFO = {'BC1179': {'start': 58,
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


CASES_TO_REMOVE_TXT = """BNB1172(DF)
/content/drive/MyDrive/Tesi/Dataset-arrays/BNB1172(DF)/images/BNB1172(DF)_130.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/BNB1172(DF)/images/BNB1172(DF)_131.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/BNB1172(DF)/images/BNB1172(DF)_196.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/BNB1172(DF)/images/BNB1172(DF)_197.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/BNB1172(DF)/images/BNB1172(DF)_198.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/BNB1172(DF)/images/BNB1172(DF)_199.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/BNB1172(DF)/images/BNB1172(DF)_200.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/BNB1172(DF)/images/BNB1172(DF)_201.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/BNB1172(DF)/images/BNB1172(DF)_202.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/BNB1172(DF)/images/BNB1172(DF)_203.npy

D1AP5(VR)
/content/drive/MyDrive/Tesi/Dataset-arrays/ D1AP5(VR)/images/D1AP5(VR)_91.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/ D1AP5(VR)/images/D1AP5(VR)_92.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/ D1AP5(VR)/images/D1AP5(VR)_97.npy

D1AP7(VR)
/content/drive/MyDrive/Tesi/Dataset-arrays/D1AP7(VR)/images/D1AP7(VR)_56.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D1AP7(VR)/images/D1AP7(VR)_78.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D1AP7(VR)/images/D1AP7(VR)_79.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D1AP7(VR)/images/D1AP7(VR)_80.npy

D1AP12(VR)
/content/drive/MyDrive/Tesi/Dataset-arrays/D1AP12(VR)/images/D1AP12(VR)_21.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D1AP12(VR)/images/D1AP12(VR)_22.npy

D2MP1(VR)
/content/drive/MyDrive/Tesi/Dataset-arrays/D2MP1(VR)/images/D2MP1(VR)_47.npy
BRUTTINO

D2MP3(VR)
/content/drive/MyDrive/Tesi/Dataset-arrays/D2MP3(VR)/images/D2MP3(VR)_133.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D2MP3(VR)/images/D2MP3(VR)_146.npy
C’è massa non segmentata, probabilmente benigna ma chiarire

D2MP4(VR)
/content/drive/MyDrive/Tesi/Dataset-arrays/D2MP4(VR)/images/D2MP4(VR)_60.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D2MP4(VR)/images/D2MP4(VR)_61.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D2MP4(VR)/images/D2MP4(VR)_84.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D2MP4(VR)/images/D2MP4(VR)_86.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D2MP4(VR)/images/D2MP4(VR)_128.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D2MP4(VR)/images/D2MP4(VR)_129.npy

D2MP6(VR)
/content/drive/MyDrive/Tesi/Dataset-arrays/D2MP6(VR)/images/D2MP6(VR)_24.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D2MP6(VR)/images/D2MP6(VR)_31.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D2MP6(VR)/images/D2MP6(VR)_35.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D2MP6(VR)/images/D2MP6(VR)_52.npy

D3MP7 (VR)
/content/drive/MyDrive/Tesi/Dataset-arrays/D3MP7 (VR)/images/D3MP7 (VR)_38.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D3MP7 (VR)/images/D3MP7 (VR)_39.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D3MP7 (VR)/images/D3MP7 (VR)_40.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D3MP7 (VR)/images/D3MP7 (VR)_41.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D3MP7 (VR)/images/D3MP7 (VR)_42.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D3MP7 (VR)/images/D3MP7 (VR)_64.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D3MP7 (VR)/images/D3MP7 (VR)_65.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D3MP7 (VR)/images/D3MP7 (VR)_66.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D3MP7 (VR)/images/D3MP7 (VR)_67.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D3MP7 (VR)/images/D3MP7 (VR)_68.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D3MP7 (VR)/images/D3MP7 (VR)_69.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D3MP7 (VR)/images/D3MP7 (VR)_70.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D3MP7 (VR)/images/D3MP7 (VR)_71.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D3MP7 (VR)/images/D3MP7 (VR)_72.npy

DFC1168(DF)
/content/drive/MyDrive/Tesi/Dataset-arrays/DFC1168(DF)/images/DFC1168(DF)_173.npy

DTM0772(1,5)
/content/drive/MyDrive/Tesi/Dataset-arrays/DTM0772(1,5)/images/DTM0772(1,5)_21.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/DTM0772(1,5)/images/DTM0772(1,5)_47.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/DTM0772(1,5)/images/DTM0772(1,5)_50.npy

GMG0961(3)
/content/drive/MyDrive/Tesi/Dataset-arrays/GMG0961(3)/images/GMG0961(3)_71.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/GMG0961(3)/images/GMG0961(3)_81.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/GMG0961(3)/images/GMG0961(3)_82.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/GMG0961(3)/images/GMG0961(3)_83.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/GMG0961(3)/images/GMG0961(3)_87.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/GMG0961(3)/images/GMG0961(3)_101.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/GMG0961(3)/images/GMG0961(3)_102.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/GMG0961(3)/images/GMG0961(3)_104.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/GMG0961(3)/images/GMG0961(3)_109.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/GMG0961(3)/images/GMG0961(3)_110.npy
LGM0159(1,5)
/content/drive/MyDrive/Tesi/Dataset-arrays/LGM0159(1,5)/images/LGM0159(1,5)_41.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/LGM0159(1,5)/images/LGM0159(1,5)_42.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/LGM0159(1,5)/images/LGM0159(1,5)_43.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/LGM0159(1,5)/images/LGM0159(1,5)_44.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/LGM0159(1,5)/images/LGM0159(1,5)_45.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/LGM0159(1,5)/images/LGM0159(1,5)_46.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/LGM0159(1,5)/images/LGM0159(1,5)_47.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/LGM0159(1,5)/images/LGM0159(1,5)_48.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/LGM0159(1,5)/images/LGM0159(1,5)_49.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/LGM0159(1,5)/images/LGM0159(1,5)_70.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/LGM0159(1,5)/images/LGM0159(1,5)_71.npy
Bruttino

MD0773(DF)
/content/drive/MyDrive/Tesi/Dataset-arrays/MD0773(DF)/images/MD0773(DF)_115.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/MD0773(DF)/images/MD0773(DF)_116.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/MD0773(DF)/images/MD0773(DF)_153.npy

PF0473(1,5)
/content/drive/MyDrive/Tesi/Dataset-arrays/PF0473(1,5)/images/PF0473(1,5)_50.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PF0473(1,5)/images/PF0473(1,5)_51.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PF0473(1,5)/images/PF0473(1,5)_52.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PF0473(1,5)/images/PF0473(1,5)_59.npy
Brutto

PMG0761(3)
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_73.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_85.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_89.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_90.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_91.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_92.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_93.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_93.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_95.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_96.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_97.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_98.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_99.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_112.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_113.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_124.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_125.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_126.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_133.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_134.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_135.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_136.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_137.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_145.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_146.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_147.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_148.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_149.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_150.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_151.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_152.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_153.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PMG0761(3)/images/PMG0761(3)_154.npy
Bruttissimo

RD0175
/content/drive/MyDrive/Tesi/Dataset-arrays/RD0175/images/RD0175_82.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/RD0175/images/RD0175_83.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/RD0175/images/RD0175_123.npy

SM0972(DF)
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_104.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_126.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_127.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_128.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_129.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_135.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_142.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_143.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_144.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_145.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_146.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_147.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_148.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_149.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_150.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_151.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_152.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_153.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_154.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_155.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_156.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_157.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_158.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_159.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_160.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_161.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_162.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_163.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM0972(DF)/images/SM0972(DF)_164.npy
Usare in validation

UFR0987
/content/drive/MyDrive/Tesi/Dataset-arrays/UFR0987/images/UFR0987_71.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/UFR0987/images/UFR0987_72.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/UFR0987/images/UFR0987_73.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/UFR0987/images/UFR0987_89.npy

ZT0279(3)
/content/drive/MyDrive/Tesi/Dataset-arrays/ZT0279(3)/images/ZT0279(3)_137.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/ZT0279(3)/images/ZT0279(3)_138.npy

BC1179B
/content/drive/MyDrive/Tesi/Dataset-arrays/BC1179B-merged/images/BC1179B_57.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/BC1179B-merged/images/BC1179B_71.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/BC1179B-merged/images/BC1179B_96.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/BC1179B-merged/images/BC1179B_97.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/BC1179B-merged/images/BC1179B_115.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/BC1179B-merged/images/BC1179B_116.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/BC1179B-merged/images/BC1179B_117.npy

D2MP9b(VR)-merged
/content/drive/MyDrive/Tesi/Dataset-arrays/D2MP9b(VR) merged/images/D2MP9b(VR)_35.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D2MP9b(VR) merged/images/D2MP9b(VR)_36.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D2MP9b(VR) merged/images/D2MP9b(VR)_37.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D2MP9b(VR)-merged/images/D2MP9b(VR)_45.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D2MP9b(VR)-merged/images/D2MP9b(VR)_63.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/D2MP9b(VR)-merged/images/D2MP9b(VR)_69.npy

GA07(DF)B-merged
/content/drive/MyDrive/Tesi/Dataset-arrays/GA07(DF)B-merged/images/GA07(DF)B_62.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/GA07(DF)B-merged/images/GA07(DF)B_83.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/GA07(DF)B-merged/images/GA07(DF)B_104.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/GA07(DF)B-merged/images/GA07(DF)B_105.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/GA07(DF)B-merged/images/GA07(DF)B_106.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/GA07(DF)B-merged/images/GA07(DF)B_119.npy

DCC0340(1,5)-merged
/content/drive/MyDrive/Tesi/Dataset-arrays/DCC0340(1,5)-merged/images/DCC0340(1,5)_27.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/DCC0340(1,5)-merged/images/DCC0340(1,5)_28.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/DCC0340(1,5)-merged/images/DCC0340(1,5)_29.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/DCC0340(1,5)-merged/images/DCC0340(1,5)_30.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/DCC0340(1,5)-merged/images/DCC0340(1,5)_31.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/DCC0340(1,5)-merged/images/DCC0340(1,5)_32.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/DCC0340(1,5)-merged/images/DCC0340(1,5)_33.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/DCC0340(1,5)-merged/images/DCC0340(1,5)_46.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/DCC0340(1,5)-merged/images/DCC0340(1,5)_47.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/DCC0340(1,5)-merged/images/DCC0340(1,5)_48.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/DCC0340(1,5)-merged/images/DCC0340(1,5)_49.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/DCC0340(1,5)-merged/images/DCC0340(1,5)_50.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/DCC0340(1,5)-merged/images/DCC0340(1,5)_51.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/DCC0340(1,5)-merged/images/DCC0340(1,5)_55.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/DCC0340(1,5)-merged/images/DCC0340(1,5)_56.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/DCC0340(1,5)-merged/images/DCC0340(1,5)_57.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/DCC0340(1,5)-merged/images/DCC0340(1,5)_58.npy

HV1263(1,5)-merged
/content/drive/MyDrive/Tesi/Dataset-arrays/HV1263(1,5)-merged/images/HV1263(1,5)_5.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/HV1263(1,5)-merged/images/HV1263(1,5)_6.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/HV1263(1,5)-merged/images/HV1263(1,5)_7.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/HV1263(1,5)-merged/images/HV1263(1,5)_17.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/HV1263(1,5)-merged/images/HV1263(1,5)_18.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/HV1263(1,5)-merged/images/HV1263(1,5)_19.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/HV1263(1,5)-merged/images/HV1263(1,5)_20.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/HV1263(1,5)-merged/images/HV1263(1,5)_21.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/HV1263(1,5)-merged/images/HV1263(1,5)_22.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/HV1263(1,5)-merged/images/HV1263(1,5)_23.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/HV1263(1,5)-merged/images/HV1263(1,5)_24.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/HV1263(1,5)-merged/images/HV1263(1,5)_21.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/HV1263(1,5)-merged/images/HV1263(1,5)_35.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/HV1263(1,5)-merged/images/HV1263(1,5)_36.npy

LA0248B-merged
/content/drive/MyDrive/Tesi/Dataset-arrays/LA0248B-merged/images/LA0248B_115.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/LA0248B-merged/images/LA0248B_120.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/LA0248B-merged/images/LA0248B_128.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/LA0248B-merged/images/LA0248B_146.npy

PE0468(1,5)B-merged
/content/drive/MyDrive/Tesi/Dataset-arrays/PE0468(1,5)B-merged/images/PE0468(1,5)B_17.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PE0468(1,5)B-merged/images/PE0468(1,5)B_18.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PE0468(1,5)B-merged/images/PE0468(1,5)B_19.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PE0468(1,5)B-merged/images/PE0468(1,5)B_28.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PE0468(1,5)B-merged/images/PE0468(1,5)B_29.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PE0468(1,5)B-merged/images/PE0468(1,5)B_30.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PE0468(1,5)B-merged/images/PE0468(1,5)B_38.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PE0468(1,5)B-merged/images/PE0468(1,5)B_48.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PE0468(1,5)B-merged/images/PE0468(1,5)B_49.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PE0468(1,5)B-merged/images/PE0468(1,5)B_50.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PE0468(1,5)B-merged/images/PE0468(1,5)B_51.npy

MV1276(1,5)B-merged
/content/drive/MyDrive/Tesi/Dataset-arrays/MV1276(1,5)B-merged/images/MV1276(1,5)B_41.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/MV1276(1,5)B-merged/images/MV1276(1,5)B_42.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/MV1276(1,5)B-merged/images/MV1276(1,5)B_43.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/MV1276(1,5)B-merged/images/MV1276(1,5)B_44.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/MV1276(1,5)B-merged/images/MV1276(1,5)B_45.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/MV1276(1,5)B-merged/images/MV1276(1,5)B_46.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/MV1276(1,5)B-merged/images/MV1276(1,5)B_58.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/MV1276(1,5)B-merged/images/MV1276(1,5)B_59.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/MV1276(1,5)B-merged/images/MV1276(1,5)B_60.npy

VS0976(1,5)B-merged
/content/drive/MyDrive/Tesi/Dataset-arrays/VS0976(1,5)B-merged/images/VS0976(1,5)B_20.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VS0976(1,5)B-merged/images/VS0976(1,5)B_21.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VS0976(1,5)B-merged/images/VS0976(1,5)B_22.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VS0976(1,5)B-merged/images/VS0976(1,5)B_27.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VS0976(1,5)B-merged/images/VS0976(1,5)B_28.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VS0976(1,5)B-merged/images/VS0976(1,5)B_29.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VS0976(1,5)B-merged/images/VS0976(1,5)B_36.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VS0976(1,5)B-merged/images/VS0976(1,5)B_37.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VS0976(1,5)B-merged/images/VS0976(1,5)B_43.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VS0976(1,5)B-merged/images/VS0976(1,5)B_44.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VS0976(1,5)B-merged/images/VS0976(1,5)B_45.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VS0976(1,5)B-merged/images/VS0976(1,5)B_46.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VS0976(1,5)B-merged/images/VS0976(1,5)B_47.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VS0976(1,5)B-merged/images/VS0976(1,5)B_55.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VS0976(1,5)B-merged/images/VS0976(1,5)B_58.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VS0976(1,5)B-merged/images/VS0976(1,5)B_59.npy
BRUTTO

VDMB0751(DF)B-merged
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_74.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_75.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_78.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_85.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_86.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_87.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_88.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_89.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_90.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_91.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_92.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_93.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_94.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_95.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_96.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_97.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_98.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_99.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_100.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_101.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_102.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_103.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_104.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_105.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_106.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_107.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_108.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_109.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_110.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_111.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_112.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_130.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_136.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_146.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_147.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_148.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_149.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_152.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/VDMB0751(DF)B-merged/images/VDMB0751(DF)B_157.npy
VALIDATION

SM1232B-merged
/content/drive/MyDrive/Tesi/Dataset-arrays/SM1232B-merged/images/SM1232B_84.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM1232B-merged/images/SM1232B_85.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM1232B-merged/images/SM1232B_86.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/SM1232B-merged/images/SM1232B_138.npy

PS0446(1.5)B-merged
/content/drive/MyDrive/Tesi/Dataset-arrays/PS0446(1.5)B-merged/images/PS0446(1.5)B_28.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PS0446(1.5)B-merged/images/PS0446(1.5)B_29.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PS0446(1.5)B-merged/images/PS0446(1.5)B_34.npy
/content/drive/MyDrive/Tesi/Dataset-arrays/PS0446(1.5)B-merged/images/PS0446(1.5)B_40.npy"""

# Patients to exclude 
PATIENTS_TO_EXCLUDE = [
    'CMG0948',
    'IM0544(1,5)B-merged',
    'D2MP8(VR)',
    "VC0285(1,5)B-merged"
]

def convert_cases_to_remove_txt(text: str) -> Dict[str, List[str]]:
    """Convert cases to remove text to dictionary."""
    lines = text.strip().split("\n")
    case_dict = {}
    current_case = None

    for line in lines:
        if line == "":
            continue
        if line.startswith('/'):
            if current_case:
                case_dict[current_case].append("Dataset-arrays-4-FINAL/" + "/".join(line.split('/')[-3:]))
        else:
            current_case = line.strip()
            case_dict[current_case] = []

    case_dict = {k: v for k, v in case_dict.items() if v}
    
    return case_dict

TO_REMOVE_DICT = convert_cases_to_remove_txt(CASES_TO_REMOVE_TXT)

def get_filenames(suffix: str, base_path: str, patient_ids: List[str], 
                 remove_black_samples: bool = False, 
                 get_random_samples_and_remove_black_samples: bool = False, 
                 get_top_bottom_and_remove_black_samples: bool = False,
                 random_samples_indexes_list: Optional[List] = None, 
                 remove_picked_samples: bool = False) -> Tuple[List[str], Optional[List]]:
    """
    Get filenames with exact filtering logic from reference notebook.
    
    Args:
        suffix: Directory suffix (e.g., "images", "masks")
        base_path: Base path to dataset
        patient_ids: List of patient IDs
        remove_black_samples: Whether to apply sample-aware filtering
        get_random_samples_and_remove_black_samples: Whether to get random samples
        get_top_bottom_and_remove_black_samples: Whether to get top/bottom samples
        random_samples_indexes_list: List of random sample indexes
        remove_picked_samples: Whether to remove picked samples
        
    Returns:
        Tuple of (filenames, random_samples_indexes_list)
    """
    filenames = []

    create_random_samples_index_list = False

    if random_samples_indexes_list is None:
        random_samples_indexes_list = []
        create_random_samples_index_list=True

    for idx, patient_id in enumerate(patient_ids):
        path = os.path.join(base_path, patient_id) + "/" + suffix + "/"
        files = [os.path.join(path, p) for p in natsorted(os.listdir(path), alg=ns.IGNORECASE)]

        if get_random_samples_and_remove_black_samples:
              files_sampled = filter_samples_sample_aware(files, patient_id)
              if remove_picked_samples:
                  files_sampled = filter_samples_to_exclude(files_sampled, patient_id)
              filenames += files_sampled
              size = int(len(files_sampled)*0.25)


              random_samples_indexes = None if create_random_samples_index_list else random_samples_indexes_list[idx]
              files_random, random_samples_indexes = get_samples_size(files=files, patient_id=patient_id, size=size, random_samples=True, random_samples_indexes=random_samples_indexes)

              if create_random_samples_index_list:
                  random_samples_indexes_list.append(random_samples_indexes)

              filenames += files_random

        elif remove_black_samples:
              files_sampled = filter_samples_sample_aware(files, patient_id)
              if remove_picked_samples:
                  files_sampled = filter_samples_to_exclude(files_sampled, patient_id)
                
              filenames += files_sampled

        elif get_top_bottom_and_remove_black_samples:
              files_sampled = filter_samples_sample_aware(files, patient_id)
              if remove_picked_samples:
                  files_sampled = filter_samples_to_exclude(files_sampled, patient_id)
              filenames += files_sampled

              size = int(len(files_sampled)*0.25)
              files_top_bottom = get_samples_size(files=files, patient_id=patient_id, size=size, random_samples=False)
              filenames += files_top_bottom

        else:
              filenames += files

    if get_random_samples_and_remove_black_samples:
      return filenames, random_samples_indexes_list
    else:
      return filenames, None

def get_samples_size(files: List[str], patient_id: str, size: int, 
                    random_samples: bool = True, 
                    random_samples_indexes: Optional[List] = None):
    """Get samples of specific size from files (exact copy from reference notebook)."""
    
    if patient_id not in PATIENT_INFO:
        # If patient not in dictionary, return empty
        return [], None if random_samples else []
    
    start, end = PATIENT_INFO[patient_id]['start'], PATIENT_INFO[patient_id]['end']
    
    # Get slices outside the mass range  
    top_slices = files[:start]  
    bottom_slices = files[end:] 
    
    sample_size_top_slices = size
    sample_size_bottom_slices = size
    
    if random_samples:
        if random_samples_indexes:
            subset_top_slices_random_indexes = random_samples_indexes[0]
        else:
            if sample_size_top_slices > len(top_slices):
                sample_size_top_slices = len(top_slices)
            subset_top_slices_random_indexes = random.sample(range(len(top_slices)), sample_size_top_slices)
            
        subset_top_slices = [top_slices[i] for i in subset_top_slices_random_indexes]

        if random_samples_indexes:
            subset_bottom_slices_random_indexes = random_samples_indexes[1]
        else:
            if sample_size_bottom_slices > len(bottom_slices):
                sample_size_bottom_slices = len(bottom_slices)
            subset_bottom_slices_random_indexes = random.sample(range(len(bottom_slices)), sample_size_bottom_slices)
            
        subset_bottom_slices = [bottom_slices[i] for i in subset_bottom_slices_random_indexes]

        files_to_return = subset_top_slices + subset_bottom_slices
        return files_to_return, [subset_top_slices_random_indexes, subset_bottom_slices_random_indexes]

    else:
        if sample_size_top_slices > len(top_slices):
            sample_size_top_slices = len(top_slices)
        if sample_size_bottom_slices > len(bottom_slices):
            sample_size_bottom_slices = len(bottom_slices)

        subset_top_slices = top_slices[-sample_size_top_slices:]
        subset_bottom_slices = bottom_slices[:sample_size_bottom_slices]
        files_to_return = subset_top_slices + subset_bottom_slices
        return files_to_return

def filter_samples_sample_aware(files: List[str], patient_id: str) -> List[str]:
    """Filter samples based on patient-specific start/end (exact copy from reference notebook)."""
    if patient_id not in PATIENT_INFO:
        return files
    
    start, end = PATIENT_INFO[patient_id]['start'], PATIENT_INFO[patient_id]['end']
    return files[start+1:end]

def filter_samples_to_exclude(files: List[str], patient_id: str) -> List[str]:
    """Filter out samples that should be excluded (exact copy from reference notebook)."""
    if patient_id not in TO_REMOVE_DICT:
        return files
    
    files_to_exclude = TO_REMOVE_DICT[patient_id]
    filtered_list = []
    
    for file in files:
        file_clean = file.replace("mask_", "")
        file_clean = file_clean.replace("masks", "images")
        
        if file_clean not in files_to_exclude:
            filtered_list.append(file)

    return filtered_list

def get_train_val_test_dicts(dataset_base_path: str, x_train: List[str], 
                            x_val: List[str], x_test: List[str]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Get train/val/test dictionaries using exact approach from reference notebook.
    
    Args:
        dataset_base_path: Base path to dataset
        x_train: Training patient IDs
        x_val: Validation patient IDs  
        x_test: Test patient IDs
        
    Returns:
        Tuple of (train_dicts, val_dicts, test_dicts)
    """
    # Get training data
    train_images_fnames, _ = get_filenames(
        suffix="images", 
        base_path=dataset_base_path, 
        patient_ids=x_train, 
        remove_black_samples=False,
        get_top_bottom_and_remove_black_samples=True,
        random_samples_indexes_list=None, 
        remove_picked_samples=True
    )
    train_masks_fnames, _ = get_filenames(
        suffix="masks", 
        base_path=dataset_base_path,
        patient_ids=x_train, 
        remove_black_samples=False,
        get_top_bottom_and_remove_black_samples=True,
        random_samples_indexes_list=None, 
        remove_picked_samples=True
    )
    
    val_images_fnames, _ = get_filenames(
        suffix="images", 
        base_path=dataset_base_path,
        patient_ids=x_val, 
        remove_black_samples=False,
        get_random_samples_and_remove_black_samples=False,
        random_samples_indexes_list=None, 
        remove_picked_samples=False
    )
    val_masks_fnames, _ = get_filenames(
        suffix="masks", 
        base_path=dataset_base_path,
        patient_ids=x_val, 
        remove_black_samples=False,
        get_random_samples_and_remove_black_samples=False,
        random_samples_indexes_list=None, 
        remove_picked_samples=False
    )
    
    test_images_fnames, _ = get_filenames(
        suffix="images", 
        base_path=dataset_base_path,
        patient_ids=x_test, 
        remove_black_samples=False,
        get_random_samples_and_remove_black_samples=False,
        random_samples_indexes_list=None, 
        remove_picked_samples=False
    )
    test_masks_fnames, _ = get_filenames(
        suffix="masks", 
        base_path=dataset_base_path,
        patient_ids=x_test, 
        remove_black_samples=False,
        get_random_samples_and_remove_black_samples=False,
        random_samples_indexes_list=None, 
        remove_picked_samples=False
    )
    
    # Create data dictionaries
    train_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images_fnames, train_masks_fnames)]
    val_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(val_images_fnames, val_masks_fnames)]
    test_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(test_images_fnames, test_masks_fnames)]
    
    return train_dicts, val_dicts, test_dicts
