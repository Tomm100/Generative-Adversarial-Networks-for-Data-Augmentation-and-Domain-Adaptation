"""
Pipeline DANN per Synthetic Domain Adaptation (Synth→Real).

Obiettivo: mitigare il "Synthetic Domain Shift" rilevato tramite analisi
PCA/t-SNE. Il DANN forza il feature extractor a essere cieco alla differenza
reale/sintetica, costringendo la rete a imparare solo la patologia.

Pipeline:
  STEP 0 — Generazione Immagini Sintetiche NORMAL (da GAN epoch 300)
  STEP 1 — Baseline: ResNet standard (Phase 1 da main.py) sul Test Set reale
  STEP 2 — Training DANN (Synth→Real domain adaptation)
  STEP 3 — Valutazione DANN sul Test Set reale
  STEP 4 — Confronto PRE vs POST DANN

Prerequisiti:
  - main.py già eseguito (checkpoint ResNet Phase 1 disponibile)
  - GAN checkpoint disponibile in results/gan_checkpoints/
  - Dataset in ./data/modified_dataset/

Utilizzo:
  python main_dann_synth.py
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
from models.dann_synth import DANNSynth
from models.resnet import ResNetClassifier
from models.wgan import Generator
from train_dann_synth import train_dann_synth, evaluate_dann_synth
from eval import evaluate_on_test, generate_synthetic_images
from utils.seed import set_seed

# ==============================================================================
# ⚙️ CONFIGURAZIONE
# ==============================================================================
# Epoch della GAN da usare per generare le immagini sintetiche
GAN_EPOCH_TO_USE = 220

# Quante immagini NORMAL sintetiche generare (bilanciare col train reale)
# Metti 0 per ricalcolare automaticamente dal gap NORMAL/PNEUMONIA
NUM_SYNTHETIC_NORMAL = 0  # 0 = auto (verrà calcolato dal gap del training set)

# Configurazione DANN
DANN_EPOCHS      = 50
DANN_LR_FEAT     = 1e-4   # LR basso per preservare i pesi pretrained
DANN_LR_CLASS    = 1e-3   # LR alto per i classificatori
DANN_BETA1       = 0.5    # β₁ Adam ridotto per stabilità con GRL
DANN_BATCH       = 32
DANN_ALPHA_SYNTH = 0.5    # peso della task loss sui sintetici (Supervised DA)
                          # 0.0 = UDA puro (sintetici ignorati dalla testa classif.)
                          # 0.5 = bilanciato (reale domina, sintetico supplementare)
                          # 1.0 = simmetrico (reale = sintetico)

DANN_CKPT_DIR = "./results/dann_synth_checkpoints"
# ==============================================================================


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*60}")
    print(f"  DANN Synth→Real Pipeline — Device: {device}")
    print(f"{'═'*60}")

    # Setup dataset
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        return
    train_dir, val_dir, test_dir = res

    # Calcolo gap da colmare (se non specificato manualmente)
    n_normal   = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    n_pneum    = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    num_synth  = NUM_SYNTHETIC_NORMAL if NUM_SYNTHETIC_NORMAL > 0 else (n_pneum - n_normal)
    print(f"\n  Train reale: {n_normal} NORMAL + {n_pneum} PNEUMONIA")
    print(f"  Immagini sintetiche da generare: {num_synth} NORMAL")

    # ── STEP 0: Generazione Immagini Sintetiche ──────────────────────────────
    gan_ckpt = os.path.join("/content/drive/MyDrive/ProgettoMLVM/results_SNGAN/sngan_checkpoints/", f"G_epoch_{GAN_EPOCH_TO_USE}.pth")
    if not os.path.isfile(gan_ckpt):
        print(f"\n  ❌ Checkpoint GAN non trovato: {gan_ckpt}")
        print(f"  Verifica che main.py sia già stato eseguito e il training completato.")
        return

    syn_normal_dir = os.path.join(SYNTHETIC_DIR, 'NORMAL')
    # Genera solo se non già presenti (per evitare di rigenerare ad ogni run)
    if not os.path.exists(syn_normal_dir) or len(os.listdir(syn_normal_dir)) < num_synth:
        print(f"\n  [STEP 0] Generazione {num_synth} immagini sintetiche NORMAL...")
        G = Generator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=GAN_D).to(device)
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
    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        name=f"DANN_Synth_ep{GAN_EPOCH_TO_USE}",
        config={
            "phase":            "DANN_Synth_Domain_Adaptation",
            "seed":             SEED,
            "gan_epoch":        GAN_EPOCH_TO_USE,
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

    # ── STEP 1: Baseline (ResNet Phase 1 su Test Set reale) ──────────────────
    print(f"\n{'═'*60}")
    print(f"  STEP 1: Baseline — ResNet Phase 1 sul Test Set reale")
    print(f"{'═'*60}")

    resnet_ckpt = os.path.join(CHECKPOINTS_DIR, 'best_model_Phase1.pth')
    if not os.path.isfile(resnet_ckpt):
        print(f"  ⚠️  Checkpoint Phase 1 non trovato: {resnet_ckpt}")
        print(f"  Esegui prima main.py per generarlo. Salto la baseline.")
        report_baseline = None
        cm_baseline     = None
    else:
        _, _, test_loader_base, class_names = get_dataloaders(
            train_dir, val_dir, test_dir,
            img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE
        )
        resnet_model = ResNetClassifier(num_classes=2)
        report_baseline, cm_baseline = evaluate_on_test(
            resnet_model, resnet_ckpt, test_loader_base, class_names, device,
            tag="Baseline_Phase1", out_dir=METRICS_DIR
        )
        print(f"  Baseline NORMAL Recall: {report_baseline['NORMAL']['recall']:.4f}")
        del resnet_model

    # ── STEP 2: DataLoader DANN ───────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  STEP 2: Setup DataLoader DANN Synth→Real")
    print(f"{'═'*60}")

    (source_loader, target_loader,
     val_loader, test_loader, class_names) = get_synth_dann_loaders(
        real_train_dir=train_dir,
        synthetic_normal_dir=syn_normal_dir,
        real_val_dir=val_dir,
        real_test_dir=test_dir,
        img_size=RESNET_IMG_SIZE,
        batch_size=DANN_BATCH
    )

    # ── STEP 3: Training DANN ─────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  STEP 3: Training DANN Synth→Real")
    print(f"{'═'*60}")

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

    # ── STEP 4: Valutazione DANN sul Test Set reale ───────────────────────────
    print(f"\n{'═'*60}")
    print(f"  STEP 4: Valutazione DANN — Test Set REALE")
    print(f"{'═'*60}")

    report_dann, cm_dann = evaluate_dann_synth(
        model=model,
        ckpt_path=ckpt_path,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        tag="DANN_Synth",
        out_dir=METRICS_DIR
    )

    # ── STEP 5: Confronto finale ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CONFRONTO: Baseline vs DANN Synth→Real")
    print(f"{'='*60}")

    if report_baseline is not None:
        for cls in class_names:
            for metric in ['precision', 'recall', 'f1-score']:
                v_base = report_baseline[cls][metric]
                v_dann = report_dann[cls][metric]
                diff   = v_dann - v_base
                arrow  = "↑" if diff > 0 else "↓"
                print(f"  {cls} {metric:12s}: {v_base:.4f} → {v_dann:.4f}  ({arrow} {abs(diff):.4f})")

        acc_b = report_baseline['accuracy']
        acc_d = report_dann['accuracy']
        print(f"\n  Overall Acc: {acc_b:.4f} → {acc_d:.4f}  "
              f"({'↑' if acc_d > acc_b else '↓'} {abs(acc_d - acc_b):.4f})")
    
    print(f"\n  💾 Checkpoint DANN: {ckpt_path}")
    print(f"  ✅ Pipeline DANN Synth→Real completata!")

    wandb.finish()


if __name__ == '__main__':
    main()
