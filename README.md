# VENUS: Breast Segmentation Framework

## Introduction

[TODO: Add introduction and paper details]

## Data Preparation

### Data Preprocessing

[TODO: Add data preprocessing details specific to your datasets]

### Structure of Data Folder

```
venus/                               # Project root
├── breast_segmentation/             # Backend functions & architectures (including VENUS)
├── run-scripts/                     # Command-line training and testing scripts
│   ├── train/                       # Training scripts
│   │   ├── train_baselines_breadm.py
│   │   ├── train_baselines_private.py
│   │   ├── train_venus_breadm.py
│   │   └── train_venus_private.py
│   └── test/                        # Inference/testing scripts
│       ├── inference_dataset_aware_breadm.py
│       └── inference_dataset_aware_private.py
├── run-notebooks/                   # Jupyter notebook counterparts
│   ├── train/                       # Training notebooks
│   └── test/                        # Testing notebooks
├── checkpoints/                     # Model checkpoints
│   ├── breadm-dataset/
│   └── private-dataset/
├── download/                        # Dataset download utilities
├── BreaDM/                          # BreaDM dataset (gitignored)
│   └── seg/                         # Segmentation data
└── Dataset-arrays-4-FINAL/         # Private dataset (gitignored)
```

## Implementation

### Hardware Prerequisites

[TODO: Add hardware specifications]

### Dependencies

- Python 3.8+
- CUDA-compatible GPU recommended
- Install dependencies using Poetry:

```bash
git clone [repository-url]
cd venus
poetry install
poetry shell
```

### Configuration

All model parameters, hyperparameters, dataset paths, and training configurations are specified in `breast_segmentation/config/settings.py`. Modify this file to adjust:

- Dataset paths
- Model architectures
- Training hyperparameters
- Preprocessing parameters
- Checkpoint directories

### Training

#### Command Line Scripts

Train baseline models on BreaDM dataset:
```bash
poetry run python run-scripts/train/train_baselines_breadm.py
```

Train baseline models on private dataset:
```bash
poetry run python run-scripts/train/train_baselines_private.py
```

Train VENUS model on BreaDM dataset:
```bash
poetry run python run-scripts/train/train_venus_breadm.py
```

Train VENUS model on private dataset:
```bash
poetry run python run-scripts/train/train_venus_private.py
```

#### Jupyter Notebooks

For detailed pipeline visualization and step-by-step execution, use the corresponding notebooks in `run-notebooks/train/`:

- `training_baselines_breastdm.ipynb`
- `training_baselines_private.ipynb`
- `training_venus_breastdm.ipynb`
- `training_venus_private.ipynb`

### Testing

#### Command Line Scripts

Run inference on BreaDM dataset:
```bash
poetry run python run-scripts/test/inference_dataset_aware_breadm.py
```

Run inference on private dataset:
```bash
poetry run python run-scripts/test/inference_dataset_aware_private.py
```

#### Jupyter Notebooks

For detailed inference analysis, use the corresponding notebooks in `run-notebooks/test/`:

- `inference_dataset_aware_breastdm.ipynb`
- `inference_dataset_aware_private.ipynb`

### Model Weights

[TODO: Add model weights information and download links]

## Citation

[TODO: Add citation information]

## Acknowledgments

[TODO: Add acknowledgments]

## Contact

[TODO: Add contact information]
