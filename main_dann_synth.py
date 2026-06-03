"""Pipeline DANN per Synthetic Domain Adaptation (Synth->Real)."""


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
from models.dann_synth import DANNSynth
from eval import evaluate_on_test, generate_synthetic_images
from train_dann_synth import train_dann_synth, evaluate_dann_synth
from utils.seed import set_seed

# ==============================================================================
# CONFIGURAZIONE GAN (Decommentare UNA sola configurazione alla volta)
# ==============================================================================


GAN_EPOCH_TO_USE = 220


# 1. WGAN-GP 128 con PatchGAN + BAGAN
# from models.wgan import Generator
# GAN_CKPT_DIR_GAN = "/content/drive/MyDrive/ProgettoMLVM/results_WGAN_Pg_Bg_128/gan_checkpoints"
# GEN_D = 128
#
# 2. WGAN-GP 128 senza PatchGAN e senza BAGAN
# from models.wgan import Generator
# GAN_CKPT_DIR_GAN = "/content/drive/MyDrive/ProgettoMLVM/results_WGAN_noPg_noBg_128/gan_checkpoints"
# GEN_D = 128
#
# 3. SNGAN 128 con PatchGAN + BAGAN
# NOTA: i checkpoint PG+BG 128 sono addestrati con architettura wgan.py Generator
from models.wgan import Generator
GAN_CKPT_DIR_GAN = "/content/drive/MyDrive/ProgettoMLVM/results_SNGAN_pg_bg_128/sngan_checkpoints"
GEN_D = 128
#
# 4. SNGAN 128 senza PatchGAN e senza BAGAN
# from models.sngan_128 import SNGenerator as Generator
# GAN_CKPT_DIR_GAN = "/content/drive/MyDrive/ProgettoMLVM/results_SNGAN_noPg_noBg_128/sngan_checkpoints"
# GEN_D = 128
#
# 5. SNGAN 256 con PatchGAN + BAGAN
# from models.sngan import SNGenerator as Generator
# GAN_CKPT_DIR_GAN = "/content/drive/MyDrive/ProgettoMLVM/results_SNGAN_pg_bg_256/sngan_checkpoints"
# GEN_D = 128
#
# 6. SNGAN 256 senza PatchGAN e senza BAGAN
# from models.sngan import SNGenerator as Generator
# GAN_CKPT_DIR_GAN = "/content/drive/MyDrive/ProgettoMLVM/results_SNGAN_noPg_noBg_256/sngan_checkpoints"
# GEN_D = 128
#
#
GAN_WEIGHTS_PATH = os.path.join(GAN_CKPT_DIR_GAN, f"G_epoch_{GAN_EPOCH_TO_USE}.pth")

# ==============================================================================
# CONFIGURAZIONE GENERALE
# ==============================================================================

SYNTH_GAP_PERCENT = 100

NUM_SYNTHETIC_NORMAL = 0


DANN_EPOCHS      = 50
DANN_LR_FEAT     = 1e-4
DANN_LR_CLASS    = 1e-3
DANN_BETA1       = 0.5
DANN_BATCH       = 32
DANN_ALPHA_SYNTH = 0.5


DANN_CKPT_DIR = "./results/dann_synth_checkpoints"
# ==============================================================================


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  DANN Synth->Real Pipeline -- Device: {device}")
    print(f"{'='*60}")


    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        return
    train_dir, val_dir, test_dir = res


    n_normal   = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    n_pneum    = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    max_deficit = n_pneum - n_normal

    if NUM_SYNTHETIC_NORMAL > 0:
        num_synth = NUM_SYNTHETIC_NORMAL
    else:
        num_synth = int(max_deficit * (SYNTH_GAP_PERCENT / 100.0))

    print(f"Train reale: {n_normal} NORMAL + {n_pneum} PNEUMONIA")
    print(f"Gap totale: {max_deficit} immagini")
    print(f"Gap da colmare: {SYNTH_GAP_PERCENT}% -> {num_synth} immagini sintetiche NORMAL")

    gan_ckpt = GAN_WEIGHTS_PATH
    if not os.path.isfile(gan_ckpt):
        print(f"Checkpoint GAN non trovato: {gan_ckpt}")
        print(f"Verifica il path e l'epoch configurati.")
        return

    syn_normal_dir = os.path.join(SYNTHETIC_DIR, 'NORMAL')

    if not os.path.exists(syn_normal_dir) or len(os.listdir(syn_normal_dir)) < num_synth:
        print(f"\n  [STEP 0] Generazione {num_synth} immagini sintetiche NORMAL...")
        G = Generator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=GEN_D).to(device)
        G.load_state_dict(torch.load(gan_ckpt, map_location=device))
        generate_synthetic_images(
            G, num_gen_normal=num_synth, num_gen_pneumonia=0,
            nz=GAN_NZ, n_class=GAN_N_CLASS, device=device,
            syn_dir=SYNTHETIC_DIR
        )
        del G
        torch.cuda.empty_cache()
    else:
        print(f"\n  [STEP 0] Immagini sintetiche già presenti ({len(os.listdir(syn_normal_dir))} img). Skip.")


    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        name=f"DANN_Synth_ep{GAN_EPOCH_TO_USE}_gap{SYNTH_GAP_PERCENT}pct",
        config={
            "phase":            "DANN_Synth_Domain_Adaptation",
            "seed":             SEED,
            "gan_epoch":        GAN_EPOCH_TO_USE,
            "gan_weights":      GAN_WEIGHTS_PATH,
            "synth_gap_pct":    SYNTH_GAP_PERCENT,
            "num_synth":        num_synth,
            "dann_epochs":      DANN_EPOCHS,
            "dann_lr_feat":     DANN_LR_FEAT,
            "dann_lr_class":    DANN_LR_CLASS,
            "dann_beta1":       DANN_BETA1,
            "dann_batch":       DANN_BATCH,
            "dann_alpha_synth": DANN_ALPHA_SYNTH,
            "img_size":         RESNET_IMG_SIZE,
        }
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)


    print(f"\n{'='*60}")
    print(f"  STEP 1: Setup DataLoader DANN Synth->Real")
    print(f"{'='*60}")

    (source_loader, target_loader,
     val_loader, test_loader, class_names) = get_synth_dann_loaders(
        real_train_dir=train_dir,
        synthetic_normal_dir=syn_normal_dir,
        real_val_dir=val_dir,
        real_test_dir=test_dir,
        img_size=RESNET_IMG_SIZE,
        batch_size=DANN_BATCH
    )


    print(f"\n{'='*60}")
    print(f"  STEP 2: Training DANN Synth->Real")
    print(f"{'='*60}")

    model = DANNSynth(num_classes=2, pretrained=True)
    model, history, ckpt_path = train_dann_synth(
        model=model,
        source_loader=source_loader,
        target_loader=target_loader,
        device=device,
        epochs=DANN_EPOCHS,
        lr_feature=DANN_LR_FEAT,
        lr_classifier=DANN_LR_CLASS,
        beta1=DANN_BETA1,
        alpha_synth=DANN_ALPHA_SYNTH,
        tag="DANN_Synth",
        checkpoints_dir=DANN_CKPT_DIR,
        val_loader=val_loader,
        class_names=class_names
    )


    print(f"\n{'='*60}")
    print(f"  STEP 3: Valutazione DANN -- Test Set REALE")
    print(f"{'='*60}")

    report_dann, cm_dann = evaluate_dann_synth(
        model=model,
        ckpt_path=ckpt_path,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        tag="DANN_Synth",
        out_dir=METRICS_DIR
    )


    print(f"\n{'='*60}")
    print(f"  RISULTATI DANN Synth->Real")
    print(f"{'='*60}")
    for cls in class_names:
        for metric in ['precision', 'recall', 'f1-score']:
            v = report_dann[cls][metric]
            print(f"  {cls} {metric:12s}: {v:.4f}")
    print(f"\n  Overall Accuracy: {report_dann['accuracy']:.4f}")
    print(f"\n  Checkpoint DANN: {ckpt_path}")
    print(f"  Pipeline DANN Synth->Real completata!")

    wandb.finish()


if __name__ == '__main__':
    main()
