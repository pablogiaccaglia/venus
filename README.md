# Venus - Medical Image Segmentation Project

This project implements medical image segmentation using various deep learning approaches, including boundary loss techniques for improved segmentation accuracy.

## Features

- Medical image segmentation using MONAI framework
- Boundary loss implementation for enhanced edge detection
- Multiple neural network architectures (UNet, SwinUNETR, etc.)
- Comprehensive data processing and augmentation pipelines
- Integration with Weights & Biases for experiment tracking

## Project Structure

```
├── boundaryloss/           # Boundary loss implementation
│   ├── dataloader.py      # Custom data loading utilities
│   ├── losses.py          # Loss function implementations
│   └── utils.py           # Utility functions
├── utiils.py              # Additional utility functions
├── training_*.ipynb       # Training notebooks
├── inference_*.ipynb      # Inference notebooks
└── pyproject.toml         # Poetry configuration
```

## Setup

This project uses Poetry for dependency management. Follow these steps to set up the environment:

1. Install Poetry (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Usage

After setting up the environment, you can run the Jupyter notebooks for training and inference:

```bash
jupyter lab
```

## Dependencies

- PyTorch & MONAI for deep learning
- OpenCV & scikit-image for image processing
- Albumentations for data augmentation
- Weights & Biases for experiment tracking
- Lightning for training framework

## License

[Add your license information here]
