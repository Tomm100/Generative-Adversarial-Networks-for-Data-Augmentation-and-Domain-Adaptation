# GAN-Augmented Chest X-Ray Classification & Domain Adaptation

## Project Overview — Machine Learning for Vision and Multimedia

### Objective
This project investigates the effectiveness of **GAN-based data augmentation** and **Domain Adaptation** techniques for improving the classification of chest X-ray images into two categories: **Normal** and **Pneumonia**. 

Medical imaging datasets often suffer from severe class imbalance. This project addresses this issue by:
1. Generating high-quality synthetic training images of the minority class using advanced Generative Adversarial Networks (WGAN-GP and SNGAN).
2. Employing **Domain-Adversarial Neural Networks (DANN)** to mitigate the "Synthetic Domain Shift" between real and GAN-generated images. This forces the feature extractor to learn pathology-specific features that are invariant to the source domain (real vs. synthetic), thus maximizing the utility of the synthetic data.

### Dataset
- **Source**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes**: NORMAL and PNEUMONIA.
- **Challenge**: The dataset is naturally imbalanced. We synthesize NORMAL images to balance the training distribution.

### Models and Architectures
- **Classifiers**:
  - `ResNet-18` (pre-trained on ImageNet) adapted for binary classification.
- **Generative Models**:
  - `WGAN-GP` (Wasserstein GAN with Gradient Penalty).
  - `SNGAN` (Spectral Normalization GAN with Hinge Loss).
  - Both GANs feature conditional generation, optionally enhanced with **PatchGAN** discriminators and **BAGAN** (Balancing GAN) class-conditioning initialization.
- **Domain Adaptation**:
  - `DANNSynth`: A Domain-Adversarial Neural Network built on a ResNet-18 backbone. It uses a **Gradient Reversal Layer (GRL)** to align the feature distributions of real (source) and synthetic (target) images via adversarial training.

### Pipeline and Scripts
The project is modularized into several main scripts to execute different experimental pipelines:

- `main.py`: Full pipeline using **WGAN-GP**. Trains the baseline classifier, trains the WGAN-GP, generates synthetic images, and trains the GAN-augmented ResNet classifier.
- `main_sngan.py`: Full pipeline using **SNGAN**.
- `main_baselines.py`: Evaluates standard data-balancing techniques without GANs (**Class Weighting** and **Oversampling** via `WeightedRandomSampler`).
- `main_dann_synth.py`: Pipeline for **Supervised Synthetic Domain Adaptation (Synth→Real)** using DANN. It mitigates the domain shift by training a domain discriminator concurrently with the label predictor.
- `main_evaluate_gan_epochs.py`: Evaluates the fidelity (FID and KID metrics) of GAN checkpoints over time using `torch-fidelity`.

### Key Features
- **Weights & Biases (W&B) Integration**: Comprehensive logging of losses, evaluation metrics (Macro F1, Accuracy, Precision, Recall), confusion matrices, and generated synthetic samples.
- **Dynamic GRL Scheduling**: The DANN implementation uses a dynamically scheduled adaptation parameter ($\lambda$) based on Ganin et al. (2016) for stable adversarial training.
- **Automated Dataset Setup**: Built-in routines to safely extract, restructure, and prepare the dataset with proper train/val/test splits.

### Project Structure
```
├── dataset/                    # Dataset loaders, WeightedRandomSampler, and preparation scripts
├── models/                     # Model architectures (wgan.py, sngan.py, resnet.py, dann_synth.py)
├── utils/                      # Utilities for logging, visualization, and seeding
├── config.py                   # Centralized hyperparameters and path configurations
├── train.py                    # Training loops for GANs and baseline ResNet
├── train_dann_synth.py         # Training and evaluation loop for DANN
├── eval.py                     # Evaluation routines and synthetic data generation
├── main.py                     # Entry point for WGAN-GP pipeline
├── main_sngan.py               # Entry point for SNGAN pipeline
├── main_dann_synth.py          # Entry point for DANN Domain Adaptation
├── main_baselines.py           # Entry point for standard baseline comparisons
└── main_evaluate_gan_epochs.py # Entry point for GAN fidelity evaluation
```