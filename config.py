"""Configurazione centralizzata del progetto."""

import os

# Paths: Input
DATA_DIR        = "./data"
DATASET_DIR     = os.path.join(DATA_DIR, "modified_dataset")

# Paths: Output
RESULTS_DIR         = "./results"
METRICS_DIR         = os.path.join(RESULTS_DIR, "metrics")
GAN_SAMPLES_DIR     = os.path.join(RESULTS_DIR, "gan_samples")
GAN_CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "gan_checkpoints")
SYNTHETIC_DIR       = os.path.join(RESULTS_DIR, "synthetic_images")
CHECKPOINTS_DIR     = os.path.join(RESULTS_DIR, "checkpoints")
AUGMENTED_DIR       = os.path.join(RESULTS_DIR, "augmented_dataset")

# ResNet
RESNET_IMG_SIZE    = 224
RESNET_BATCH_SIZE  = 32
RESNET_EPOCHS      = 10
RESNET_LR          = 1e-4
RESNET_NUM_CLASSES = 2

# WGAN-GP
GAN_IMG_SIZE              = 128
GAN_BATCH_SIZE            = 64
GAN_EPOCHS                = 300
GAN_LR                    = 0.0001
GAN_BETA1                 = 0.0
GAN_BETA2                 = 0.9
GAN_D_WEIGHT_DECAY        = 1e-3
GAN_EPSILON_PENALTY_COEFF = 1e-3
GAN_N_CRITIC              = 5
GAN_NZ                    = 100
GAN_N_CLASS               = 2
GAN_NC                    = 1
GAN_D                     = 128
GAN_SAVE_EVERY            = 10
GAN_DRIVE_BACKUP_EVERY    = 50
GAN_DRIVE_DIR             = "/content/drive/MyDrive/ProgettoMLVM/GAN_CHECKPOINTS_BACKUP"
GAN_WEIGHT_INIT_MEAN      = 0.0
GAN_WEIGHT_INIT_STD       = 0.02
GAN_NUM_VIS_SAMPLES       = 6
GAN_GEN_BATCH_SIZE        = 64
GAN_JPEG_QUALITY          = 95
GAN_VALIDATE_EVERY        = 50
GAN_VAL_RESNET_EPOCHS     = 5

# SNGAN
SNGAN_IMG_SIZE     = 128
SNGAN_EPOCHS       = 300
SNGAN_LR           = 1e-4
SNGAN_N_CRITIC     = 1
SNGAN_D            = 128
SNGAN_BATCH_SIZE   = 64
SNGAN_SAVE_EVERY   = 10
SNGAN_SAMPLES_DIR  = os.path.join(RESULTS_DIR, "sngan_samples")
SNGAN_CKPT_DIR     = os.path.join(RESULTS_DIR, "sngan_checkpoints")
SNGAN_SYNTH_DIR    = os.path.join(RESULTS_DIR, "sngan_synthetic_images")
SNGAN_AUG_DIR      = os.path.join(RESULTS_DIR, "sngan_augmented_dataset")

# DANN
DANN_SOURCE_DIR      = os.path.join(DATA_DIR, "source_domain")
DANN_TARGET_DIR      = os.path.join(DATA_DIR, "target_domain")
DANN_IMG_SIZE        = 128
DANN_BATCH_SIZE      = 32
DANN_EPOCHS          = 50
DANN_LR_FEATURE      = 1e-4
DANN_LR_CLASSIFIER   = 1e-3
DANN_BETA1           = 0.5
DANN_CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "dann_checkpoints")

# DataLoader
NUM_WORKERS        = 4
PIN_MEMORY         = True
PERSISTENT_WORKERS = True

# Riproducibilita
SEED = 42
