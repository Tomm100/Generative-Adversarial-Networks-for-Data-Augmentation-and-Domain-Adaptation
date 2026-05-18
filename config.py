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
GAN_EPOCHS      = 300
GAN_LR          = 0.0001
GAN_BETA1                = 0.0   # β₁ Adam (0.0 raccomandato per WGAN-GP)
GAN_BETA2                = 0.9   # β₂ Adam
GAN_D_WEIGHT_DECAY       = 1e-3  # Weight decay ottimizzatore Critic
GAN_EPSILON_PENALTY_COEFF = 1e-3 # Coefficiente epsilon drift penalty (evita divergenza logit Critic)
GAN_N_CRITIC             = 5
GAN_NZ                   = 100
GAN_N_CLASS              = 2
GAN_NC                   = 1
GAN_D                    = 128
GAN_SAVE_EVERY           = 10   # Salva checkpoint locali ogni N epoche
GAN_DRIVE_BACKUP_EVERY   = 50   # Copia i checkpoint su Google Drive ogni N epoche
GAN_DRIVE_DIR            = "/content/drive/MyDrive/ProgettoMLVM/GAN_CHECKPOINTS_BACKUP"  # Path Drive (Colab)
GAN_WEIGHT_INIT_MEAN     = 0.0   # Media inizializzazione pesi G e D
GAN_WEIGHT_INIT_STD      = 0.02  # Std inizializzazione pesi G e D
GAN_NUM_VIS_SAMPLES      = 6     # Immagini per classe nella griglia di visualizzazione
GAN_GEN_BATCH_SIZE       = 64    # Batch size durante la generazione di immagini sintetiche
GAN_JPEG_QUALITY         = 95    # Qualità JPEG delle immagini sintetiche salvate
# GAN_LR_MILESTONES = [60, 80]  # Non più in uso (WGAN-GP ora usa LinearLR continuo verso 0)
# GAN_LR_GAMMA    = 0.2         # Non più in uso
GAN_VALIDATE_EVERY       = 50   # Validazione ogni N epoche GAN
GAN_VAL_RESNET_EPOCHS    = 5    # Epoche ResNet ridotte per validazione periodica

# ─── SNGAN (Spectral Normalization GAN) ──────────────────
SNGAN_EPOCHS       = 300
SNGAN_LR           = 1e-4     # Stesso LR per G e D (niente TTUR)
SNGAN_N_CRITIC     = 1        # Hinge+SN non richiede n_critic=5
SNGAN_D            = 128      # Dim base G e D (come WGAN-GP: GAN_D=128)
SNGAN_SAVE_EVERY   = 10
SNGAN_SAMPLES_DIR  = os.path.join(RESULTS_DIR, "sngan_samples")
SNGAN_CKPT_DIR     = os.path.join(RESULTS_DIR, "sngan_checkpoints")
SNGAN_SYNTH_DIR    = os.path.join(RESULTS_DIR, "sngan_synthetic_images")
SNGAN_AUG_DIR      = os.path.join(RESULTS_DIR, "sngan_augmented_dataset")

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

# ─── DataLoader ──────────────────────────────────────────
NUM_WORKERS        = 4     # Worker CPU per prefetching (consigliato >= 4 su A100)
PIN_MEMORY         = True  # Trasferimento asincrono CPU→GPU via DMA (richiede CUDA)
PERSISTENT_WORKERS = True  # Mantieni i worker tra le epoche (evita re-spawn)

# ─── Riproducibilità ─────────────────────────────────────
SEED = 42
