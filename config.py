"""
Configurazione centralizzata del progetto.
Tutti gli iperparametri e i path sono definiti qui.
"""

import os

# ─── Paths: Input ────────────────────────────────────────
DATA_DIR        = "./data"
DATASET_DIR     = os.path.join(DATA_DIR, "modified_dataset")

# ─── Paths: Output ───────────────────────────────────────
CHECKPOINTS_DIR   = "./checkpoints"
RESULTS_DIR       = "./results"
GAN_SAMPLES_DIR   = "./gan_training_samples"
GAN_CHECKPOINTS_DIR = os.path.join(CHECKPOINTS_DIR, "gan")
SYNTHETIC_DIR     = "./synthetic_images"
AUGMENTED_DIR     = "./augmented_dataset"

# ─── ResNet ──────────────────────────────────────────────
RESNET_IMG_SIZE    = 128
RESNET_BATCH_SIZE  = 32
RESNET_EPOCHS      = 10
RESNET_LR          = 0.001
RESNET_NUM_CLASSES = 2

# ─── WGAN-GP ─────────────────────────────────────────────
GAN_IMG_SIZE    = 128
GAN_BATCH_SIZE  = 64
GAN_EPOCHS      = 100
GAN_LR          = 0.0001
GAN_N_CRITIC    = 5
GAN_NZ          = 100
GAN_N_CLASS     = 2
GAN_NC          = 1
GAN_D           = 128
GAN_SAVE_EVERY  = 20
GAN_LR_DECAY    = {60: 5, 80: 5}

# ─── Riproducibilità ─────────────────────────────────────
SEED = 42
