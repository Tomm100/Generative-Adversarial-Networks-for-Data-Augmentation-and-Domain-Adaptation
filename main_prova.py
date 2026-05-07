"""
main_prova.py — Script di esperimento rapido.

Riusa i pesi GAN già salvati (senza riaddestrare il GAN).
Rigenera le immagini sintetiche come JPEG e rilancia solo Phase 1 + Phase 3
per verificare se il cambio di formato elimina lo shortcut learning.

Uso:
    python main_prova.py                           # usa il checkpoint migliore (epoca 50)
    python main_prova.py --gan_epoch 80            # specifica un'epoca diversa
    python main_prova.py --gan_epoch best           # cerca il miglior checkpoint disponibile
"""

import torch
import os
import shutil
import argparse
import wandb

from config import (
    DATASET_DIR, RESULTS_DIR, METRICS_DIR,
    RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_EPOCHS, RESNET_LR,
    GAN_NZ, GAN_N_CLASS, GAN_NC, GAN_D,
    SEED,
)
from dataset.loader import setup_dataset, get_dataloaders
from models.wgan import Generator
from eval import evaluate_on_test, generate_synthetic_images, plot_comparison
from train import train_resnet
from utils.seed import set_seed


# Path dei pesi GAN dal run precedente (cartella rinominata dall'utente)
PREV_GAN_CHECKPOINTS_DIR = "./results_with_norm/gan_checkpoints"

# Directory dedicate per questo esperimento (non sovrascrive i risultati originali)
EXP_DIR = os.path.join(RESULTS_DIR, "experiment_jpeg")
EXP_SYNTHETIC_DIR = os.path.join(EXP_DIR, "synthetic_images")
EXP_AUGMENTED_DIR = os.path.join(EXP_DIR, "augmented_dataset")
EXP_METRICS_DIR = os.path.join(EXP_DIR, "metrics")
EXP_CHECKPOINTS_DIR = os.path.join(EXP_DIR, "checkpoints")


def find_best_checkpoint(models_dir):
    """Trova il checkpoint GAN con l'epoca più alta disponibile."""
    if not os.path.exists(models_dir):
        return None, 0
    checkpoints = [f for f in os.listdir(models_dir) if f.startswith('G_epoch_') and f.endswith('.pth')]
    if not checkpoints:
        return None, 0
    epochs = [int(f.replace('G_epoch_', '').replace('.pth', '')) for f in checkpoints]
    best_epoch = max(epochs)
    return os.path.join(models_dir, f'G_epoch_{best_epoch}.pth'), best_epoch


def main():
    parser = argparse.ArgumentParser(description="Esperimento rapido: JPEG fix + ImageNet norm")
    parser.add_argument('--gan_epoch', type=str, default='50',
                        help='Epoca del checkpoint GAN da usare (numero o "best")')
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  ESPERIMENTO: JPEG fix + ImageNet normalization")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # --- W&B ---
    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        name="experiment_jpeg_fix",
        config={
            "experiment": "jpeg_fix",
            "seed": SEED,
            "resnet_img_size": RESNET_IMG_SIZE,
            "resnet_batch_size": RESNET_BATCH_SIZE,
            "resnet_epochs": RESNET_EPOCHS,
            "resnet_lr": RESNET_LR,
            "gan_d": GAN_D,
            "normalization": "imagenet",
            "synthetic_format": "jpeg",
        }
    )

    # --- 1. SETUP DATASET ---
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        return
    train_dir, val_dir, test_dir = res

    n_train_n = len([f for f in os.listdir(os.path.join(train_dir, 'NORMAL')) if not f.startswith('.')])
    n_train_p = len([f for f in os.listdir(os.path.join(train_dir, 'PNEUMONIA')) if not f.startswith('.')])
    print(f"  Train count: {n_train_n} NORMAL + {n_train_p} PNEUMONIA")

    num_gen_normal = n_train_p - n_train_n
    num_gen_pneumonia = 0

    # --- 2. DATALOADERS (ImageNet normalization) ---
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(EXP_METRICS_DIR, exist_ok=True)

    # --- 3. PHASE 1: RESNET BASELINE (solo dati reali, ImageNet norm) ---
    print(f"\n{'='*60}")
    print(f"  Phase 1: Baseline ResNet (ImageNet norm)")
    print(f"{'='*60}")

    # Sovrascrivi CHECKPOINTS_DIR temporaneamente per non sporcare
    import config
    original_checkpoints = config.CHECKPOINTS_DIR
    config.CHECKPOINTS_DIR = EXP_CHECKPOINTS_DIR

    model_p1, hist_p1, ckpt_p1 = train_resnet(
        train_loader, val_loader, device,
        epochs=RESNET_EPOCHS, lr=RESNET_LR, tag="Exp_Phase1")

    report_p1, cm_p1 = evaluate_on_test(
        model_p1, ckpt_p1, test_loader, classes, device,
        tag="Exp_Phase1", out_dir=EXP_METRICS_DIR)

    # --- 4. CARICA PESI GAN E RIGENERA SINTETICHE COME JPEG ---
    if args.gan_epoch == 'best':
        gan_ckpt_path, gan_epoch = find_best_checkpoint(PREV_GAN_CHECKPOINTS_DIR)
    else:
        gan_epoch = int(args.gan_epoch)
        gan_ckpt_path = os.path.join(PREV_GAN_CHECKPOINTS_DIR, f'G_epoch_{gan_epoch}.pth')

    if gan_ckpt_path is None or not os.path.exists(gan_ckpt_path):
        print(f"\n❌ Checkpoint GAN non trovato: {gan_ckpt_path}")
        print(f"   Checkpoints disponibili in {PREV_GAN_CHECKPOINTS_DIR}:")
        if os.path.exists(PREV_GAN_CHECKPOINTS_DIR):
            for f in sorted(os.listdir(PREV_GAN_CHECKPOINTS_DIR)):
                if f.startswith('G_'):
                    print(f"     - {f}")
        wandb.finish()
        return

    print(f"\n{'='*60}")
    print(f"  Caricamento Generator da epoca {gan_epoch}")
    print(f"  Checkpoint: {gan_ckpt_path}")
    print(f"  Le sintetiche verranno salvate come JPEG (quality=95)")
    print(f"{'='*60}")

    G = Generator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=GAN_D).to(device)
    G.load_state_dict(torch.load(gan_ckpt_path, map_location=device))
    G.eval()

    # Pulisci sintetiche precedenti dell'esperimento
    if os.path.exists(EXP_SYNTHETIC_DIR):
        shutil.rmtree(EXP_SYNTHETIC_DIR)

    generate_synthetic_images(
        G, num_gen_normal, num_gen_pneumonia,
        nz=GAN_NZ, n_class=GAN_N_CLASS, device=device,
        syn_dir=EXP_SYNTHETIC_DIR)

    # --- 5. CREA DATASET AUGMENTED ---
    aug_train_dir = os.path.join(EXP_AUGMENTED_DIR, 'train')
    if os.path.exists(EXP_AUGMENTED_DIR):
        shutil.rmtree(EXP_AUGMENTED_DIR)
    shutil.copytree(train_dir, aug_train_dir)

    for cat in ['NORMAL', 'PNEUMONIA']:
        syn_cat = os.path.join(EXP_SYNTHETIC_DIR, cat)
        if os.path.exists(syn_cat):
            for f in os.listdir(syn_cat):
                shutil.copy(os.path.join(syn_cat, f),
                            os.path.join(aug_train_dir, cat, f))

    n_aug_n = len([f for f in os.listdir(os.path.join(aug_train_dir, 'NORMAL')) if not f.startswith('.')])
    n_aug_p = len([f for f in os.listdir(os.path.join(aug_train_dir, 'PNEUMONIA')) if not f.startswith('.')])
    print(f"\n  Augmented dataset: {n_aug_n} NORMAL + {n_aug_p} PNEUMONIA")

    # --- 6. PHASE 3: RESNET AUGMENTED ---
    print(f"\n{'='*60}")
    print(f"  Phase 3: Augmented ResNet (ImageNet norm, JPEG sintetiche)")
    print(f"{'='*60}")

    aug_train_loader, _, _, _ = get_dataloaders(
        aug_train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    model_p3, hist_p3, ckpt_p3 = train_resnet(
        aug_train_loader, val_loader, device,
        epochs=RESNET_EPOCHS, lr=RESNET_LR, tag="Exp_Phase3")

    report_p3, cm_p3 = evaluate_on_test(
        model_p3, ckpt_p3, test_loader, classes, device,
        tag="Exp_Phase3", out_dir=EXP_METRICS_DIR)

    # --- 7. CONFRONTO ---
    plot_comparison(hist_p1, hist_p3, cm_p1, cm_p3, classes,
                    report_p1, report_p3, out_dir=EXP_METRICS_DIR)

    # Ripristina config originale
    config.CHECKPOINTS_DIR = original_checkpoints

    print(f"\n📊 Risultati esperimento salvati in: {EXP_DIR}")
    print("✅ Esperimento completato!")

    wandb.finish()


if __name__ == '__main__':
    main()
