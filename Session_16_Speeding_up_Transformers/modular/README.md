# English to French Translation with Encoder-Decoder 📚

## Introduction
This repository contains the code for training an English to French translation model code using the Encoder-Decoder architecture. The model is designed for use with the OPUS book translation dataset, with the goal of achieving a final loss of less than 1.8 during training.

## Code Structure 📁

The project is organized into the following directories:

### 1. `config` 🛠️
This directory contains configuration files where you can specify various hyperparameters, such as learning rates, batch sizes, and model architecture parameters. You can customize these settings to suit your experiment.

### 2. `dataset` 📂
In the `dataset` directory, you'll find everything related to data processing and data loading:
- Data preprocessing scripts for handling the OPUS dataset.
- Data modules for creating dataloaders and handling data augmentation .

### 3. `model` 🧠
The `model` directory is where the core of the project resides:
- Encoder-Decoder architecture implementation.
- Attention mechanisms and other components specific to the translation task.

### 4. `utils` 🛠️
The `utils` directory contains utility functions and helper scripts:
- Functions for run_validation, greedy_decode
- Any other helper functions used throughout the project.

### 5. `train` 🚂
The `train` script uses PyTorch Lightning (PL) to train the translation model.



