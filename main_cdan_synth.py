"""
Pipeline CDAN per Synthetic Domain Adaptation (Synth→Real).

Obiettivo: mitigare il "Synthetic Domain Shift" rilevato tramite analisi
PCA/t-SNE. La CDAN (Conditional Domain Adversarial Network) allinea le
feature condizionandole sulle predizioni di classe, usando anche un
peso basato sull'entropia (CDAN+E) per ignorare i campioni incerti.

Pipeline:
  STEP 0 — Generazione Immagini Sintetiche NORMAL (da GAN)
  STEP 1 — Setup DataLoader DANN Synth→Real
  STEP 2 — Training CDAN+E (Synth→Real domain adaptation)
  STEP 3 — Valutazione CDAN sul Test Set reale

Prerequisiti:
  - GAN checkpoint disponibile nel path specificato
  - Dataset in ./data/modified_dataset/

Utilizzo:
  python main_cdan_synth.py
"""

import torch
import os
import wandb

from config import (
    DATASET_DIR, RESULTS_DIR, METRICS_DIR, CHECKPOINTS_DIR,
    SYNTHETIC_DIR,
    RESNET_IMG_SIZE, RESNET_BATCH_SIZE,
    GAN_NZ, GAN_N_CLASS, GAN_NC, GAN_D,
    GAN_CHECKPOINTS_DIR,
    SEED,
)
from dataset.loader import setup_dataset, get_dataloaders
from dataset.synth_real_loader import get_synth_dann_loaders
from models.cdan_synth import CDANSynth
from eval import evaluate_on_test, generate_synthetic_images
from train_cdan_synth import train_cdan_synth, evaluate_cdan_synth
from utils.seed import set_seed

# ==============================================================================
# ⚙️ CONFIGURAZIONE GAN (Decommentare UNA sola configurazione alla volta)
# ==============================================================================

# Epoch della GAN da cui caricare i pesi del generatore
GAN_EPOCH_TO_USE = 220

#
# ── 1. WGAN-GP 128 con PatchGAN + BAGAN ──
# from models.wgan import Generator
# GAN_CKPT_DIR_GAN = "/content/drive/MyDrive/ProgettoMLVM/results_WGAN_Pg_Bg_128/gan_checkpoints"
# GEN_D = 128
#
# ── 2. WGAN-GP 128 senza PatchGAN e senza BAGAN ──
# from models.wgan import Generator
# GAN_CKPT_DIR_GAN = "/content/drive/MyDrive/ProgettoMLVM/results_WGAN_noPg_noBg_128/gan_checkpoints"
# GEN_D = 128
#
# ── 3. SNGAN 128 con PatchGAN + BAGAN ──
# NOTA: i checkpoint PG+BG 128 sono stati addestrati con l'architettura di wgan.py Generator
from models.wgan import Generator
GAN_CKPT_DIR_GAN = "/content/drive/MyDrive/ProgettoMLVM/results_SNGAN_pg_bg_128/sngan_checkpoints"
GEN_D = 128
#
# ── 4. SNGAN 128 senza PatchGAN e senza BAGAN ──
# from models.sngan_128 import SNGenerator as Generator
# GAN_CKPT_DIR_GAN = "/content/drive/MyDrive/ProgettoMLVM/results_SNGAN_noPg_noBg_128/sngan_checkpoints"
# GEN_D = 128
#
# ── 5. SNGAN 256 con PatchGAN + BAGAN ──
# from models.sngan import SNGenerator as Generator
# GAN_CKPT_DIR_GAN = "/content/drive/MyDrive/ProgettoMLVM/results_SNGAN_pg_bg_256/sngan_checkpoints"
# GEN_D = 128
#
# ── 6. SNGAN 256 senza PatchGAN e senza BAGAN ──
# from models.sngan import SNGenerator as Generator
# GAN_CKPT_DIR_GAN = "/content/drive/MyDrive/ProgettoMLVM/results_SNGAN_noPg_noBg_256/sngan_checkpoints"
# GEN_D = 128
#
# Path completo costruito automaticamente da GAN_EPOCH_TO_USE
GAN_WEIGHTS_PATH = os.path.join(GAN_CKPT_DIR_GAN, f"G_epoch_{GAN_EPOCH_TO_USE}.pth")

# ==============================================================================
# ⚙️ CONFIGURAZIONE GENERALE
# ==============================================================================

# Percentuale del gap NORMAL/PNEUMONIA da colmare con le sintetiche.
# Es: 100 = colma tutto il gap (bilanciamento completo)
#      50 = colma metà del gap
#      25 = aggiunge solo il 25% delle sintetiche necessarie
SYNTH_GAP_PERCENT = 100  # valore in [0, 100]

# Quante immagini NORMAL sintetiche generare (override manuale).
# Metti 0 per ricalcolare automaticamente dal gap NORMAL/PNEUMONIA
# secondo SYNTH_GAP_PERCENT.
NUM_SYNTHETIC_NORMAL = 0  # 0 = auto

# Configurazione CDAN
CDAN_EPOCHS      = 50
CDAN_LR_FEAT     = 1e-4   # LR basso per preservare i pesi pretrained
CDAN_LR_CLASS    = 1e-3   # LR alto per i classificatori
CDAN_BETA1       = 0.5    # β₁ Adam ridotto per stabilità con GRL
CDAN_BATCH       = 32
CDAN_ALPHA_SYNTH = 0.5    # peso della task loss sui sintetici (Supervised DA)
CDAN_USE_ENTROPY = True   # Abilita Entropy Conditioning (CDAN+E)

CDAN_CKPT_DIR = "./results/cdan_synth_checkpoints"
# ==============================================================================


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*60}")
    print(f"  CDAN Synth→Real Pipeline — Device: {device}")
    print(f"{'═'*60}")

    # Setup dataset
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        return
    train_dir, val_dir, test_dir = res

    # Calcolo gap da colmare
    n_normal   = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    n_pneum    = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    max_deficit = n_pneum - n_normal

    if NUM_SYNTHETIC_NORMAL > 0:
        num_synth = NUM_SYNTHETIC_NORMAL
    else:
        num_synth = int(max_deficit * (SYNTH_GAP_PERCENT / 100.0))

    print(f"\n  Train reale: {n_normal} NORMAL + {n_pneum} PNEUMONIA")
    print(f"  Gap totale: {max_deficit} immagini")
    print(f"  Gap da colmare: {SYNTH_GAP_PERCENT}% → {num_synth} immagini sintetiche NORMAL")

    # ── STEP 0: Generazione Immagini Sintetiche ──────────────────────────────
    gan_ckpt = GAN_WEIGHTS_PATH
    if not os.path.isfile(gan_ckpt):
        print(f"\n  ❌ Checkpoint GAN non trovato: {gan_ckpt}")
        print(f"  Verifica il path e l'epoch configurati.")
        return

    syn_normal_dir = os.path.join(SYNTHETIC_DIR, 'NORMAL')
    # Genera solo se non già presenti (per evitare di rigenerare ad ogni run)
    if not os.path.exists(syn_normal_dir) or len(os.listdir(syn_normal_dir)) < num_synth:
        print(f"\n  [STEP 0] Generazione {num_synth} immagini sintetiche NORMAL...")
        G = Generator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=GEN_D).to(device)
        G.load_state_dict(torch.load(gan_ckpt, map_location=device))
        generate_synthetic_images(
            G, num_gen_normal=num_synth, num_gen_pneumonia=0,
            nz=GAN_NZ, n_class=GAN_N_CLASS, device=device,
            syn_dir=SYNTHETIC_DIR
        )
        del G  # libera GPU
        torch.cuda.empty_cache()
    else:
        print(f"\n  [STEP 0] Immagini sintetiche già presenti ({len(os.listdir(syn_normal_dir))} img). Skip.")

    # WandB
    mode_str = "CDAN+E" if CDAN_USE_ENTROPY else "CDAN"
    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        name=f"CDAN_Synth_ep{GAN_EPOCH_TO_USE}_gap{SYNTH_GAP_PERCENT}pct",
        config={
            "phase":            "CDAN_Synth_Domain_Adaptation",
            "mode":             mode_str,
            "seed":             SEED,
            "gan_epoch":        GAN_EPOCH_TO_USE,
            "gan_weights":      GAN_WEIGHTS_PATH,
            "synth_gap_pct":    SYNTH_GAP_PERCENT,
            "num_synth":        num_synth,
            "cdan_epochs":      CDAN_EPOCHS,
            "cdan_lr_feat":     CDAN_LR_FEAT,
            "cdan_lr_class":    CDAN_LR_CLASS,
            "cdan_beta1":       CDAN_BETA1,
            "cdan_batch":       CDAN_BATCH,
            "cdan_alpha_synth": CDAN_ALPHA_SYNTH,
            "cdan_use_entropy": CDAN_USE_ENTROPY,
            "img_size":         RESNET_IMG_SIZE,
        }
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # ── STEP 1: DataLoader CDAN ───────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  STEP 1: Setup DataLoader CDAN Synth→Real")
    print(f"{'═'*60}")

    (source_loader, target_loader,
     val_loader, test_loader, class_names) = get_synth_dann_loaders(
        real_train_dir=train_dir,
        synthetic_normal_dir=syn_normal_dir,
        real_val_dir=val_dir,
        real_test_dir=test_dir,
        img_size=RESNET_IMG_SIZE,
        batch_size=CDAN_BATCH
    )

    # ── STEP 2: Training CDAN ─────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  STEP 2: Training {mode_str} Synth→Real")
    print(f"{'═'*60}")

    model = CDANSynth(num_classes=2, pretrained=True, use_entropy=CDAN_USE_ENTROPY)
    model, history, ckpt_path = train_cdan_synth(
        model=model,
        source_loader=source_loader,
        target_loader=target_loader,
        device=device,
        epochs=CDAN_EPOCHS,
        lr_feature=CDAN_LR_FEAT,
        lr_classifier=CDAN_LR_CLASS,
        beta1=CDAN_BETA1,
        alpha_synth=CDAN_ALPHA_SYNTH,
        use_entropy=CDAN_USE_ENTROPY,
        tag="CDAN_Synth",
        checkpoints_dir=CDAN_CKPT_DIR,
        val_loader=val_loader,
        class_names=class_names
    )

    # ── STEP 3: Valutazione CDAN sul Test Set reale ───────────────────────────
    print(f"\n{'═'*60}")
    print(f"  STEP 3: Valutazione {mode_str} — Test Set REALE")
    print(f"{'═'*60}")

    report_cdan, cm_cdan = evaluate_cdan_synth(
        model=model,
        ckpt_path=ckpt_path,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        tag="CDAN_Synth",
        out_dir=METRICS_DIR
    )

    # ── Riepilogo finale ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RISULTATI {mode_str} Synth→Real")
    print(f"{'='*60}")
    for cls in class_names:
        for metric in ['precision', 'recall', 'f1-score']:
            v = report_cdan[cls][metric]
            print(f"  {cls} {metric:12s}: {v:.4f}")
    print(f"\n  Overall Accuracy: {report_cdan['accuracy']:.4f}")
    print(f"\n  💾 Checkpoint CDAN: {ckpt_path}")
    print(f"  ✅ Pipeline {mode_str} Synth→Real completata!")

    wandb.finish()


if __name__ == '__main__':
    main()
