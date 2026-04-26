"""
Configurazione centralizzata del progetto.
Tutti gli iperparametri e i path sono definiti qui.
"""

import os

# ─── Paths: Input ────────────────────────────────────────
DATA_DIR        = "./data"
DATASET_DIR     = os.path.join(DATA_DIR, "modified_dataset")

# ─── Paths: Output (tutto sotto results/) ────────────────
RESULTS_DIR         = "./results"
METRICS_DIR         = os.path.join(RESULTS_DIR, "metrics")
GAN_SAMPLES_DIR     = os.path.join(RESULTS_DIR, "gan_samples")
GAN_CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "gan_checkpoints")
SYNTHETIC_DIR       = os.path.join(RESULTS_DIR, "synthetic_images")
CHECKPOINTS_DIR     = os.path.join(RESULTS_DIR, "checkpoints")
AUGMENTED_DIR       = os.path.join(RESULTS_DIR, "augmented_dataset")

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
GAN_D           = 64
GAN_SAVE_EVERY  = 10
GAN_LR_MILESTONES = [60, 80]
GAN_LR_GAMMA    = 0.2           # equivale a dividere per 5
GAN_VALIDATE_EVERY = 10         # validazione ogni N epoche GAN
GAN_VAL_RESNET_EPOCHS = 5       # epoche ResNet ridotte per validazione periodica

# ─── DANN (Domain Adaptation) ────────────────────────────
DANN_SOURCE_DIR      = os.path.join(DATA_DIR, "source_domain")     # Pediatrico (Source)
DANN_TARGET_DIR      = os.path.join(DATA_DIR, "target_domain")  # Adulti (Target)
DANN_IMG_SIZE        = 128          # Allineato a RESNET_IMG_SIZE per un confronto equo
DANN_BATCH_SIZE      = 32          # Per dominio (total batch = 64)
DANN_EPOCHS          = 50
DANN_LR_FEATURE      = 1e-4        # LR basso per backbone pretrained
DANN_LR_CLASSIFIER   = 1e-3        # LR alto per classificatori (10×)
DANN_BETA1           = 0.5         # β₁ ridotto per stabilità con GRL
DANN_CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "dann_checkpoints")

# ─── Riproducibilità ─────────────────────────────────────
SEED = 42
