"""
Main script per la Phase 4: Domain Adaptation con DANN.

Esegue Unsupervised Domain Adaptation tra:
  - Source: Mooney Augmented (Pediatrico, bilanciato via GAN)
  - Target: NIH Filtrato (Adulti, sbilanciato ~700P/3000N)

Utilizzo:
  python main_dann.py

I path dei dataset sono configurati in config.py:
  - DANN_SOURCE_DIR  →  ./data/Mooney_Augmented/
  - DANN_TARGET_DIR  →  ./data/NIH_Target_DA_Ready/

Entrambi devono avere la struttura:
  dataset_root/
  ├── train/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  ├── val/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  └── test/
      ├── NORMAL/
      └── PNEUMONIA/
"""

import torch
import os
import wandb

from config import (
    DATA_DIR, RESULTS_DIR, METRICS_DIR,
    DANN_SOURCE_DIR, DANN_TARGET_DIR,
    DANN_IMG_SIZE, DANN_BATCH_SIZE, DANN_EPOCHS,
    DANN_LR_FEATURE, DANN_LR_CLASSIFIER, DANN_BETA1,
    DANN_CHECKPOINTS_DIR,
    SEED,
)
from dataset.dann_loader import get_dann_dataloaders
from models.dann import DANN_Model
from train_dann import train_dann, evaluate_dann_on_target
from utils.seed import set_seed


def main():
    # ── Setup ──
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*60}")
    print(f"  DANN Pipeline — Device: {device}")
    print(f"{'═'*60}")

    # ── Verifica esistenza dataset ──
    for name, path in [("Source", DANN_SOURCE_DIR), ("Target", DANN_TARGET_DIR)]:
        if not os.path.isdir(path):
            print(f"\n  ❌ ERRORE: {name} dataset non trovato in: {path}")
            print(f"  Copia il dataset nella cartella corretta e riprova.")
            return
        print(f"  ✅ {name}: {path}")

    # ── WandB ──
    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        config={
            "phase": "Phase4_DANN_Domain_Adaptation",
            "seed": SEED,
            "dann_img_size": DANN_IMG_SIZE,
            "dann_batch_size": DANN_BATCH_SIZE,
            "dann_epochs": DANN_EPOCHS,
            "dann_lr_feature": DANN_LR_FEATURE,
            "dann_lr_classifier": DANN_LR_CLASSIFIER,
            "dann_beta1": DANN_BETA1,
        }
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. DATALOADERS ──
    print(f"\n{'─'*60}")
    print(f"  Loading datasets...")
    print(f"{'─'*60}")

    (source_loader, target_loader,
     target_val_loader, target_test_loader,
     class_names) = get_dann_dataloaders(
        source_dir=DANN_SOURCE_DIR,
        target_dir=DANN_TARGET_DIR,
        img_size=DANN_IMG_SIZE,
        batch_size=DANN_BATCH_SIZE
    )

    # ── 2. DANN MODEL ──
    model = DANN_Model(num_classes=2)
    print(f"\n  DANN Model:")
    print(f"    Feature Extractor:    ResNet-18 (ImageNet pretrained)")
    print(f"    Label Predictor:      FC(512 → 2)")
    print(f"    Domain Discriminator: MLP(512 → 256 → 128 → 1) + GRL")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Parameters:           {total_params:,} total, {trainable_params:,} trainable")

    # ── 3. TRAINING ──
    model, history, ckpt_path = train_dann(
        model, source_loader, target_loader, device,
        epochs=DANN_EPOCHS,
        lr_feature=DANN_LR_FEATURE,
        lr_classifier=DANN_LR_CLASSIFIER,
        beta1=DANN_BETA1,
        tag="DANN",
        checkpoints_dir=DANN_CHECKPOINTS_DIR,
        target_val_loader=target_val_loader,
        class_names=class_names
    )

    # ── 4. VALUTAZIONE FINALE SUL TARGET TEST SET ──
    print(f"\n{'═'*60}")
    print(f"  Valutazione finale sul Target Test Set")
    print(f"{'═'*60}")

    report, cm = evaluate_dann_on_target(
        model, ckpt_path, target_test_loader, class_names, device,
        tag="DANN", out_dir=METRICS_DIR
    )

    # ── Riepilogo ──
    print(f"\n{'═'*60}")
    print(f"  RIEPILOGO DANN")
    print(f"{'═'*60}")
    print(f"  Target Test Accuracy:  {report['accuracy']:.4f}")
    print(f"  Target Test Macro F1:  {report['macro avg']['f1-score']:.4f}")
    for cls in class_names:
        print(f"    {cls}: P={report[cls]['precision']:.4f} "
              f"R={report[cls]['recall']:.4f} "
              f"F1={report[cls]['f1-score']:.4f}")
    print(f"\n  📊 Risultati salvati in: {METRICS_DIR}")
    print(f"  💾 Checkpoint: {ckpt_path}")
    print(f"\n  ✅ DANN Pipeline completata con successo!")

    wandb.finish()


if __name__ == '__main__':
    main()
