"""
Main script per la Phase 5: Domain Adaptation avanzata con CDAN.

Pipeline:
  1. Valutazione PRE-CDAN: ResNet pretrainata (Phase 3 — Augmented) sul Target Test Set
     (uguale a main_dann.py — serve come baseline di confronto coerente)
  2. Training CDAN (Conditional Adversarial Domain Adaptation)
     - Inizializzato dai pesi Phase 3 (backbone + label predictor)
     - Conditional coupling: features × softmax probs → input 1024-dim al discriminatore
     - Entropy Conditioning: pesi per-campione sulla domain loss
  3. Valutazione POST-CDAN: CDAN sul Target Test Set
  4. Confronto PRE vs POST CDAN (loggato su WandB tramite log_dann_comparison)

Source: Mooney Augmented (Pediatrico, bilanciato via GAN)
Target: NIH Filtrato (Adulti, sbilanciato ~700P/3000N)

Utilizzo:
  python main_cdan.py

I path dei dataset sono configurati in config.py:
  - DANN_SOURCE_DIR  →  ./data/source_domain/
  - DANN_TARGET_DIR  →  ./data/target_domain/

Checkpoint richiesto (Phase 3):
  ./checkpoints/best_model_Phase3.pth

Entrambi i dataset devono avere la struttura:
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
from models.cdan import CDAN_Model
from models.resnet import ResNetClassifier
from train_cdan import train_cdan, evaluate_cdan_on_target
from eval import evaluate_on_test
from utils.seed import set_seed
from utils.logging import log_dann_comparison   # riusa il comparatore PRE/POST già esistente


# ═══════════════════════════════════════════════════════════════
# Weight Transfer: ResNet Phase 3 → CDAN_Model
# ═══════════════════════════════════════════════════════════════

def _load_phase3_weights_into_cdan(cdan_model, resnet_ckpt: str, device):
    """
    Trasferisce i pesi del checkpoint Phase 3 (ResNetClassifier) nel CDAN_Model.

    La struttura di CDAN_Model.feature_extractor è identica a
    DANN_Model.feature_extractor (nn.Sequential con gli stessi layer di ResNet-18),
    quindi la mappa degli indici è la medesima usata in main_dann.py:

      backbone.conv1   → feature_extractor[0]
      backbone.bn1     → feature_extractor[1]
      backbone.relu    → feature_extractor[2]  (no params)
      backbone.maxpool → feature_extractor[3]  (no params)
      backbone.layer1  → feature_extractor[4]
      backbone.layer2  → feature_extractor[5]
      backbone.layer3  → feature_extractor[6]
      backbone.layer4  → feature_extractor[7]
      backbone.avgpool → feature_extractor[8]  (no params)
      backbone.fc      → label_predictor

    Il domain_discriminator NON viene inizializzato dal checkpoint (è un
    componente nuovo, inizializzato casualmente da PyTorch di default).

    Args:
        cdan_model:   CDAN_Model instance (modificato in-place)
        resnet_ckpt:  path del checkpoint Phase 3 (.pth)
        device:       torch.device
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
        rest = k[len('backbone.'):]           # 'conv1.weight', 'layer1.0.conv1.weight', …
        top  = rest.split('.')[0]             # 'conv1', 'layer1', 'fc', …

        if top == 'fc':
            # backbone.fc.weight → label_predictor.weight
            lp_sd[rest[len('fc.'):]] = v      # 'weight' / 'bias'
        elif top in name_to_idx:
            tail    = rest[len(top):].lstrip('.')
            new_key = name_to_idx[top] + ('.' + tail if tail else '')
            fe_sd[new_key] = v

    missing_fe, unexpected_fe = cdan_model.feature_extractor.load_state_dict(
        fe_sd, strict=False)
    missing_lp, unexpected_lp = cdan_model.label_predictor.load_state_dict(
        lp_sd, strict=False)

    # Avvisi diagnostici per trasparenza
    if missing_fe:
        print(f"  ⚠️  feature_extractor — chiavi mancanti: {missing_fe}")
    if missing_lp:
        print(f"  ⚠️  label_predictor   — chiavi mancanti: {missing_lp}")


# ═══════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════

def main():
    # ── Setup riproducibilità e device ──
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*60}")
    print(f"  CDAN Pipeline — Device: {device}")
    print(f"{'═'*60}")

    # ── Verifica esistenza dataset ──
    for name, path in [("Source", DANN_SOURCE_DIR), ("Target", DANN_TARGET_DIR)]:
        if not os.path.isdir(path):
            print(f"\n  ❌ ERRORE: {name} dataset non trovato in: {path}")
            print(f"  Copia il dataset nella cartella corretta e riprova.")
            return
        print(f"  ✅ {name}: {path}")

    # ── Verifica checkpoint ResNet Phase 3 ──
    resnet_ckpt = os.path.join("./checkpoints", 'best_model_Phase3.pth')
    if not os.path.isfile(resnet_ckpt):
        print(f"\n  ❌ ERRORE: Checkpoint ResNet Phase 3 non trovato: {resnet_ckpt}")
        print(f"  Esegui prima la pipeline principale (main.py) per generare il checkpoint.")
        return
    print(f"  ✅ ResNet Phase3 checkpoint: {resnet_ckpt}")

    # ── WandB Initialization ──
    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        name=f"Phase5_CDAN_seed{SEED}",
        config={
            "phase":               "Phase5_CDAN_Domain_Adaptation",
            "method":              "CDAN",
            "entropy_conditioning": True,
            "seed":                SEED,
            "img_size":            DANN_IMG_SIZE,
            "batch_size":          DANN_BATCH_SIZE,
            "epochs":              DANN_EPOCHS,
            "lr_feature":          DANN_LR_FEATURE,
            "lr_classifier":       DANN_LR_CLASSIFIER,
            "beta1":               DANN_BETA1,
            "feature_dim":         512,
            "multilinear_dim":     1024,   # 512 × 2 classi
            "num_classes":         2,
            "backbone":            "ResNet-18",
            "init":                "Phase3_pretrained",
        }
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Caricamento DataLoader ──
    print(f"\n{'─'*60}")
    print(f"  Loading datasets (img_size={DANN_IMG_SIZE}px)...")
    print(f"{'─'*60}")

    (source_loader, target_train_loader,
     target_val_loader, target_test_loader,
     class_names) = get_dann_dataloaders(
        source_dir=DANN_SOURCE_DIR,
        target_dir=DANN_TARGET_DIR,
        img_size=DANN_IMG_SIZE,
        batch_size=DANN_BATCH_SIZE,
    )

    # ══════════════════════════════════════════════════════════
    #  STEP 1: VALUTAZIONE PRE-CDAN — ResNet Phase 3 su Target
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  STEP 1: Valutazione PRE-CDAN (Baseline Phase 3)")
    print(f"  ResNet Phase 3 (Augmented) → Target Test Set ({DANN_IMG_SIZE}px)")
    print(f"{'═'*60}")

    resnet_model = ResNetClassifier(num_classes=2)
    report_pre, cm_pre = evaluate_on_test(
        resnet_model, resnet_ckpt, target_test_loader, class_names, device,
        tag="PreCDAN_ResNet_Phase3", out_dir=METRICS_DIR
    )
    print(f"\n  PRE-CDAN Results:")
    print(f"    Accuracy:  {report_pre['accuracy']:.4f}")
    print(f"    Macro F1:  {report_pre['macro avg']['f1-score']:.4f}")
    for cls in class_names:
        print(f"    {cls}: P={report_pre[cls]['precision']:.4f} "
              f"R={report_pre[cls]['recall']:.4f} "
              f"F1={report_pre[cls]['f1-score']:.4f}")

    # ══════════════════════════════════════════════════════════
    #  STEP 2: CDAN TRAINING
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  STEP 2: CDAN Training (Conditional Domain Adaptation)")
    print(f"{'═'*60}")

    # Inizializzazione CDAN con pesi Phase 3
    model = CDAN_Model(num_classes=2, feature_dim=512)
    _load_phase3_weights_into_cdan(model, resnet_ckpt, device)
    del resnet_model  # libera memoria

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  CDAN Model (inizializzato da Phase 3 ResNet):")
    print(f"    Feature Extractor:    ResNet-18 (pesi Phase 3 — source-pretrained)")
    print(f"    Label Predictor:      FC(512 → 2) (pesi Phase 3 — source-pretrained)")
    print(f"    Domain Discriminator: MLP(1024 → 512 → 256 → 1) + GRL (random init)")
    print(f"    Multilinear dim:      512 × 2 = 1024")
    print(f"    Parameters:           {total_params:,} total, {trainable_params:,} trainable")

    model, history, ckpt_path = train_cdan(
        model, source_loader, target_train_loader, device,
        epochs=DANN_EPOCHS,
        lr_feature=DANN_LR_FEATURE,
        lr_classifier=DANN_LR_CLASSIFIER,
        beta1=DANN_BETA1,
        tag="CDAN",
        checkpoints_dir=DANN_CHECKPOINTS_DIR,
        target_val_loader=target_val_loader,
        class_names=class_names,
    )

    # ══════════════════════════════════════════════════════════
    #  STEP 3: VALUTAZIONE POST-CDAN sul Target Test Set
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  STEP 3: Valutazione POST-CDAN sul Target Test Set")
    print(f"{'═'*60}")

    report_post, cm_post = evaluate_cdan_on_target(
        model, ckpt_path, target_test_loader, class_names, device,
        tag="CDAN", out_dir=METRICS_DIR
    )

    # ══════════════════════════════════════════════════════════
    #  STEP 4: CONFRONTO PRE vs POST CDAN (WandB)
    # ══════════════════════════════════════════════════════════
    # Riuso di log_dann_comparison (già presente in utils/logging.py)
    # che produce tabella, bar chart e confusion matrices side-by-side.
    log_dann_comparison(report_pre, report_post, cm_pre, cm_post, class_names, METRICS_DIR)

    print(f"\n  💾 Checkpoint CDAN: {ckpt_path}")
    print(f"\n  ✅ CDAN Pipeline completata con successo!")

    wandb.finish()


if __name__ == '__main__':
    main()
