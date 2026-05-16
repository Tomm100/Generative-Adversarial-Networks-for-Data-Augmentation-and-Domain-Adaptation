"""
Main script per SNGAN: Spectral Normalization GAN con Hinge Loss.

Pipeline identica a main.py ma con:
  - Spectral Norm sul Critic (al posto di InstanceNorm + Gradient Penalty)
  - Hinge Loss (al posto di Wasserstein Loss)
  - n_critic=1 (al posto di 5)
  - Stessa architettura convoluzionale della WGAN-GP

Phase 1: Baseline ResNet (solo dati reali)
Phase 2: Training SNGAN (Hinge + SN)
Phase 3: Generazione sintetiche + Training ResNet su dataset augmented
Confronto Phase 1 vs Phase 3

Uso:
  python main_sngan.py
"""

import torch
import os
import shutil
import wandb

from config import (
    DATASET_DIR, RESULTS_DIR, METRICS_DIR,
    RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_EPOCHS, RESNET_LR,
    GAN_IMG_SIZE, GAN_BATCH_SIZE, GAN_NZ, GAN_N_CLASS, GAN_NC,
    GAN_BETA1, GAN_BETA2,
    GAN_NUM_VIS_SAMPLES,
    GAN_VALIDATE_EVERY, GAN_VAL_RESNET_EPOCHS,
    GAN_DRIVE_BACKUP_EVERY, GAN_DRIVE_DIR,
    NUM_WORKERS, PIN_MEMORY,
    SEED,
)
from dataset.loader import setup_dataset, get_dataloaders, get_gan_dataloader
from models.sngan import SNGenerator, SNCritic
from train_sngan import train_sngan
from train import train_resnet
from eval import evaluate_on_test, generate_synthetic_images, plot_comparison
from utils.seed import set_seed

# ══════════════════════════════════════════════════════════════
# ⚙️ CONFIGURAZIONE SNGAN
# ══════════════════════════════════════════════════════════════
SNGAN_EPOCHS       = 300
SNGAN_LR           = 1e-4     # Stesso LR per G e D (niente TTUR)
SNGAN_N_CRITIC     = 1        # Hinge+SN non richiede n_critic=5
SNGAN_D            = 128      # Dim base G e D (come WGAN-GP: GAN_D=128)
SNGAN_SAVE_EVERY   = 10
SNGAN_SAMPLES_DIR  = os.path.join(RESULTS_DIR, "sngan_samples")
SNGAN_CKPT_DIR     = os.path.join(RESULTS_DIR, "sngan_checkpoints")
SNGAN_SYNTH_DIR    = os.path.join(RESULTS_DIR, "sngan_synthetic_images")
SNGAN_AUG_DIR      = os.path.join(RESULTS_DIR, "sngan_augmented_dataset")
# ══════════════════════════════════════════════════════════════


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Avvio pipeline SNGAN su Device: {device}")

    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        name="SNGAN_Pipeline",
        config={
            "seed":         SEED,
            "architecture": "SNGAN",
            "loss":         "Hinge",
            "normalization":"Spectral Norm (Critic only)",
            "img_size":     GAN_IMG_SIZE,
            "batch_size":   GAN_BATCH_SIZE,
            "epochs":       SNGAN_EPOCHS,
            "lr":           SNGAN_LR,
            "n_critic":     SNGAN_N_CRITIC,
            "nz":           GAN_NZ,
        }
    )

    wandb.define_metric("Phase1/epoch")
    wandb.define_metric("Phase1/*",  step_metric="Phase1/epoch")
    wandb.define_metric("Phase3/epoch")
    wandb.define_metric("Phase3/*",  step_metric="Phase3/epoch")
    wandb.define_metric("SNGAN/Epoch")
    wandb.define_metric("SNGAN/*",   step_metric="SNGAN/Epoch")

    # ── 1. SETUP DATASET ──
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        return
    train_dir, val_dir, test_dir = res

    n_train_n = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    n_train_p = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    print(f"  Train: {n_train_n} NORMAL + {n_train_p} PNEUMONIA")
    num_gen_normal    = n_train_p - n_train_n
    num_gen_pneumonia = 0

    # ── 2. DATALOADERS ──
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    # ══════════════════════════════════════════════════════════
    #  PHASE 1: BASELINE RESNET
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  PHASE 1: Baseline ResNet (solo dati reali)\n{'='*60}")

    resnet_model, hist_p1, ckpt_p1 = train_resnet(
        train_loader, val_loader, device,
        epochs=RESNET_EPOCHS, lr=RESNET_LR, tag="Phase1")
    report_p1, cm_p1 = evaluate_on_test(
        resnet_model, ckpt_p1, test_loader, classes, device,
        tag="Phase1", out_dir=METRICS_DIR)

    # ══════════════════════════════════════════════════════════
    #  PHASE 2: TRAINING SNGAN
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  PHASE 2: Training SNGAN\n{'='*60}")

    gan_loader, _ = get_gan_dataloader(
        train_dir, img_size=GAN_IMG_SIZE, batch_size=GAN_BATCH_SIZE)

    G = SNGenerator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=SNGAN_D).to(device)
    D = SNCritic(nc=GAN_NC, n_class=GAN_N_CLASS, d=SNGAN_D).to(device)

    total_g = sum(p.numel() for p in G.parameters())
    total_d = sum(p.numel() for p in D.parameters())
    print(f"  Generator params:     {total_g:,}")
    print(f"  Critic params:        {total_d:,}")

    G, g_ckpt, best_epoch, val_hist = train_sngan(
        G, D, gan_loader, device,
        nz=GAN_NZ, n_class=GAN_N_CLASS,
        lr=SNGAN_LR,
        beta1=GAN_BETA1, beta2=GAN_BETA2,
        epochs=SNGAN_EPOCHS, n_critic=SNGAN_N_CRITIC,
        samples_dir=SNGAN_SAMPLES_DIR,
        models_dir=SNGAN_CKPT_DIR,
        save_every=SNGAN_SAVE_EVERY,
        drive_dir=GAN_DRIVE_DIR,
        drive_backup_every=GAN_DRIVE_BACKUP_EVERY,
        validate_every=GAN_VALIDATE_EVERY,
        resnet_epochs=GAN_VAL_RESNET_EPOCHS,
        train_dir=train_dir, val_dir=val_dir,
        num_gen_normal=num_gen_normal, num_gen_pneumonia=num_gen_pneumonia,
        resnet_img_size=RESNET_IMG_SIZE, resnet_batch_size=RESNET_BATCH_SIZE,
        augmented_dir=SNGAN_AUG_DIR,
    )

    # ══════════════════════════════════════════════════════════
    #  PHASE 3: RESNET SU DATASET AUGMENTED (SNGAN)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  PHASE 3: ResNet su Dataset Augmented (SNGAN)\n{'='*60}")

    # Usa checkpoint migliore TSTR se disponibile, altrimenti ultimo
    if best_epoch > 0:
        best_g = os.path.join(SNGAN_CKPT_DIR, f'G_epoch_{best_epoch}.pth')
        if os.path.exists(best_g):
            G.load_state_dict(torch.load(best_g, map_location=device))
            print(f"  Caricato G_epoch_{best_epoch}.pth (best TSTR)")

    generate_synthetic_images(
        G, num_gen_normal, num_gen_pneumonia,
        nz=GAN_NZ, n_class=GAN_N_CLASS, device=device,
        syn_dir=SNGAN_SYNTH_DIR)

    aug_train_dir = os.path.join(SNGAN_AUG_DIR, 'train')
    if os.path.exists(aug_train_dir):
        shutil.rmtree(aug_train_dir)
    shutil.copytree(train_dir, aug_train_dir)

    for cat in ['NORMAL', 'PNEUMONIA']:
        syn_cat = os.path.join(SNGAN_SYNTH_DIR, cat)
        if os.path.exists(syn_cat):
            for f in os.listdir(syn_cat):
                shutil.copy(os.path.join(syn_cat, f),
                            os.path.join(aug_train_dir, cat, f))

    n_aug_n = len(os.listdir(os.path.join(aug_train_dir, 'NORMAL')))
    n_aug_p = len(os.listdir(os.path.join(aug_train_dir, 'PNEUMONIA')))
    print(f"  Augmented: {n_aug_n} NORMAL + {n_aug_p} PNEUMONIA")

    aug_loader, _, _, _ = get_dataloaders(
        aug_train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    resnet_aug, hist_p3, ckpt_p3 = train_resnet(
        aug_loader, val_loader, device,
        epochs=RESNET_EPOCHS, lr=RESNET_LR, tag="Phase3")
    report_p3, cm_p3 = evaluate_on_test(
        resnet_aug, ckpt_p3, test_loader, classes, device,
        tag="Phase3", out_dir=METRICS_DIR)

    # ══════════════════════════════════════════════════════════
    #  CONFRONTO FINALE
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  CONFRONTO: Phase 1 (Baseline) vs Phase 3 (SNGAN Augmented)")
    print(f"{'='*60}")

    for cls in classes:
        for m in ['precision', 'recall', 'f1-score']:
            v1 = report_p1[cls][m]
            v3 = report_p3[cls][m]
            d  = v3 - v1
            print(f"  {cls} {m:12s}: {v1:.4f} → {v3:.4f}  "
                  f"({'↑' if d > 0 else '↓'} {abs(d):.4f})")

    acc1 = report_p1['accuracy']
    acc3 = report_p3['accuracy']
    print(f"\n  Overall Acc: {acc1:.4f} → {acc3:.4f}  "
          f"({'↑' if acc3 > acc1 else '↓'} {abs(acc3 - acc1):.4f})")

    wandb.finish()
    print(f"\n  ✅ Pipeline SNGAN completata!")


if __name__ == '__main__':
    main()
