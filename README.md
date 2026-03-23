# ChromatinGPT

ChromatinGPT predicts histone modification status from DNA sequence and chromatin accessibility data. Building on the [DeepHistone]([https://github.com/QData/DeepHistone](https://github.com/QijinYin/DeepHistone)) framework, it extends the original CNN architecture with transformer-based modules to capture long-range dependencies in regulatory sequences.

## Overview

Histone modifications play a central role in gene regulation, and predicting them computationally can reveal how chromatin state is encoded in sequence and accessibility signals. ChromatinGPT provides an end-to-end pipeline for this task — from raw data preprocessing through 5-fold cross-validated training to performance visualization — along with an experimental branch exploring transformer architectures.

## Requirements

- Python 3.8+
- PyTorch 1.7+
- NumPy 1.19+
- Pandas 1.2+
- scikit-learn 0.24+

## Installation

```bash
git clone https://github.com/lizabatski/ChromatinGPT.git
cd ChromatinGPT
pip install -r requirements.txt
```

## Project Structure

```
ChromatinGPT/
├── data_preprocessing/    # Scripts for preparing input features from raw data
├── experiments/           # Transformer-based architecture experiments
├── train_5fold.py         # Main training script (5-fold cross-validation)
├── inspect_npz.py         # Inspect contents of preprocessed .npz data files
├── visualize_dataset.py   # Visualize dataset distributions and feature statistics
├── performance_visualize.py  # Plot training curves and evaluation metrics
└── requirements.txt
```

## Usage

### Preprocessing

Prepare input data from raw sequence and accessibility files:

```bash
python data_preprocessing/<script_name>.py
```

### Training

Run 5-fold cross-validation:

```bash
python train_5fold.py
```

### Visualization and Inspection

Inspect preprocessed data:

```bash
python inspect_npz.py
```

Visualize dataset statistics:

```bash
python visualize_dataset.py
```

Plot model performance:

```bash
python performance_visualize.py
```

### Experiments

The `experiments/` directory contains transformer-based variants that replace or augment the original DeepHistone convolutional layers with self-attention mechanisms. See the individual experiment scripts for configuration details.

## Acknowledgments

This project builds on [DeepHistone]([https://github.com/QData/DeepHistone](https://github.com/QijinYin/DeepHistone)) (Yin et al.) and was developed as part of Honours research at McGill University.

## License

MIT
