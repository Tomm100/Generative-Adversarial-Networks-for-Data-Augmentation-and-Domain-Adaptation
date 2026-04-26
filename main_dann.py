"""
Main script per la Phase 4: Domain Adaptation con DANN.

Pipeline:
  1. Valutazione PRE-DANN: ResNet pretrainata (Phase 3 — Augmented) sul Target Test Set
  2. Training DANN (Unsupervised Domain Adaptation)
  3. Valutazione POST-DANN: DANN sul Target Test Set
  4. Confronto PRE vs POST Domain Adaptation (su WandB)

Source: Mooney Augmented (Pediatrico, bilanciato via GAN)
Target: NIH Filtrato (Adulti, sbilanciato ~700P/3000N)

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
    RESULTS_DIR, METRICS_DIR,
    DANN_SOURCE_DIR, DANN_TARGET_DIR,
    DANN_IMG_SIZE, DANN_BATCH_SIZE, DANN_EPOCHS,
    DANN_LR_FEATURE, DANN_LR_CLASSIFIER, DANN_BETA1,
    DANN_CHECKPOINTS_DIR,
    CHECKPOINTS_DIR,
    SEED,
)
from dataset.dann_loader import get_dann_dataloaders
from models.dann import DANN_Model
from models.resnet import ResNetClassifier
from train_dann import train_dann, evaluate_dann_on_target
from eval import evaluate_on_test
from utils.seed import set_seed
from utils.logging import log_dann_comparison


def _load_phase3_weights_into_dann(dann_model, resnet_ckpt, device):
    """
    Trasferisce i pesi del checkpoint Phase 3 (ResNetClassifier) nel DANN_Model.

    Il ResNetClassifier ha chiavi tipo: 'backbone.conv1.weight'
    Il DANN feature_extractor (nn.Sequential) ha chiavi tipo: '0.weight'

    Mappa:
      backbone.conv1   → 0
      backbone.bn1     → 1
      backbone.relu    → 2  (no params)
      backbone.maxpool → 3  (no params)
      backbone.layer1  → 4
      backbone.layer2  → 5
      backbone.layer3  → 6
      backbone.layer4  → 7
      backbone.avgpool → 8  (no params)
      backbone.fc      → label_predictor
    """
    backbone_layer_names = [
        'conv1', 'bn1', 'relu', 'maxpool',
        'layer1', 'layer2', 'layer3', 'layer4', 'avgpool'
    ]
    name_to_idx = {name: str(i) for i, name in enumerate(backbone_layer_names)}

    phase3_sd = torch.load(resnet_ckpt, map_location=device)
    fe_sd, lp_sd = {}, {}

    for k, v in phase3_sd.items():
        if not k.startswith('backbone.'):
            continue
        rest = k[len('backbone.'):]          # 'conv1.weight', 'layer1.0.conv1.weight', 'fc.weight'
        top  = rest.split('.')[0]            # 'conv1', 'layer1', 'fc', ...

        if top == 'fc':
            lp_sd[rest[len('fc.'):]] = v     # 'weight' / 'bias'
        elif top in name_to_idx:
            tail    = rest[len(top):].lstrip('.')
            new_key = name_to_idx[top] + ('.' + tail if tail else '')
            fe_sd[new_key] = v

    dann_model.feature_extractor.load_state_dict(fe_sd)
    dann_model.label_predictor.load_state_dict(lp_sd)


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

    # ── Verifica checkpoint ResNet Phase 3 ──
    resnet_ckpt = os.path.join(CHECKPOINTS_DIR, 'best_model_Phase3.pth')
    if not os.path.isfile(resnet_ckpt):
        print(f"\n  ❌ ERRORE: Checkpoint ResNet Phase 3 non trovato: {resnet_ckpt}")
        print(f"  Esegui prima la pipeline principale (main.py) per generare il checkpoint.")
        return
    print(f"  ✅ ResNet Phase3 checkpoint: {resnet_ckpt}")

    # ── WandB ──
    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        config={
            "phase":             "Phase4_DANN_Domain_Adaptation",
            "seed":              SEED,
            "dann_img_size":     DANN_IMG_SIZE,
            "dann_batch_size":   DANN_BATCH_SIZE,
            "dann_epochs":       DANN_EPOCHS,
            "dann_lr_feature":   DANN_LR_FEATURE,
            "dann_lr_classifier":DANN_LR_CLASSIFIER,
            "dann_beta1":        DANN_BETA1,
        }
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Caricamento unico dei DataLoader (tutti a DANN_IMG_SIZE = 128px) ──
    print(f"\n{'─'*60}")
    print(f"  Loading datasets (img_size={DANN_IMG_SIZE}px)...")
    print(f"{'─'*60}")

    (source_loader, target_train_loader,
     target_val_loader, target_test_loader,
     class_names) = get_dann_dataloaders(
        source_dir=DANN_SOURCE_DIR,
        target_dir=DANN_TARGET_DIR,
        img_size=DANN_IMG_SIZE,
        batch_size=DANN_BATCH_SIZE
    )

    # ══════════════════════════════════════════════════════════
    #  STEP 1: VALUTAZIONE PRE-DANN — ResNet Phase 3 su Target
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  STEP 1: Valutazione PRE-DANN")
    print(f"  ResNet Phase 3 (Augmented) → Target Test Set ({DANN_IMG_SIZE}px)")
    print(f"{'═'*60}")

    resnet_model = ResNetClassifier(num_classes=2)
    report_pre, cm_pre = evaluate_on_test(
        resnet_model, resnet_ckpt, target_test_loader, class_names, device,
        tag="PreDANN_ResNet_Phase3", out_dir=METRICS_DIR
    )
    print(f"\n  PRE-DANN Results:")
    print(f"    Accuracy:  {report_pre['accuracy']:.4f}")
    print(f"    Macro F1:  {report_pre['macro avg']['f1-score']:.4f}")
    for cls in class_names:
        print(f"    {cls}: P={report_pre[cls]['precision']:.4f} "
              f"R={report_pre[cls]['recall']:.4f} "
              f"F1={report_pre[cls]['f1-score']:.4f}")

    # ══════════════════════════════════════════════════════════
    #  STEP 2: DANN TRAINING
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  STEP 2: DANN Training (Domain Adaptation)")
    print(f"{'═'*60}")

    # Inizializzazione DANN dai pesi Phase 3 (source-pretrained):
    # il backbone porta già feature discriminative per la task su Source,
    # il DANN si concentra esclusivamente sul domain alignment.
    model = DANN_Model(num_classes=2)
    _load_phase3_weights_into_dann(model, resnet_ckpt, device)
    del resnet_model  # libera memoria GPU/CPU

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  DANN Model (inizializzato da Phase 3 ResNet):")
    print(f"    Feature Extractor:    ResNet-18 (pesi source-pretrained — Phase 3)")
    print(f"    Label Predictor:      FC(512 → 2) (pesi source-pretrained — Phase 3)")
    print(f"    Domain Discriminator: MLP(512 → 256 → 128 → 1) + GRL (random init)")
    print(f"    Parameters:           {total_params:,} total, {trainable_params:,} trainable")

    model, history, ckpt_path = train_dann(
        model, source_loader, target_train_loader, device,
        epochs=DANN_EPOCHS,
        lr_feature=DANN_LR_FEATURE,
        lr_classifier=DANN_LR_CLASSIFIER,
        beta1=DANN_BETA1,
        tag="DANN",
        checkpoints_dir=DANN_CHECKPOINTS_DIR,
        target_val_loader=target_val_loader,
        class_names=class_names
    )

    # ══════════════════════════════════════════════════════════
    #  STEP 3: VALUTAZIONE POST-DANN sul Target Test Set
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  STEP 3: Valutazione POST-DANN sul Target Test Set")
    print(f"{'═'*60}")

    report_post, cm_post = evaluate_dann_on_target(
        model, ckpt_path, target_test_loader, class_names, device,
        tag="DANN", out_dir=METRICS_DIR
    )

    # ══════════════════════════════════════════════════════════
    #  STEP 4: CONFRONTO PRE vs POST DANN (tutto su WandB)
    # ══════════════════════════════════════════════════════════
    log_dann_comparison(report_pre, report_post, cm_pre, cm_post, class_names, METRICS_DIR)

    print(f"\n  💾 Checkpoint DANN: {ckpt_path}")
    print(f"\n  ✅ DANN Pipeline completata con successo!")

    wandb.finish()


if __name__ == '__main__':
    main()
