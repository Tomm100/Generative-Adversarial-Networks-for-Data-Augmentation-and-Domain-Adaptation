"""
main_baselines.py
=================
Orchestratore per i due esperimenti baseline classici contro lo sbilanciamento,
da confrontare con la pipeline GAN (main.py).

Esperimento A — Baseline Class Weights
    • Usa il dataset originale sbilanciato (no oversampling, no GAN).
    • Penalizza la classe maggioritaria via nn.CrossEntropyLoss(weight=...).
    • I pesi sono calcolati come: w_i = N_total / (n_classes * count_i)
      (formula standard da sklearn class_weight='balanced').

Esperimento B — Baseline Oversampling
    • Usa il dataset originale sbilanciato.
    • Sovracampiona le immagini *reali* minoritarie via WeightedRandomSampler
      (nessuna immagine GAN, nessun peso alla loss).
    • Riusa get_balanced_train_dataloader() da dataset/loader.py.

Confronto su WandB
    • Project:  "gan-chest-xray-augmentation"
    • Entity:   "MachineLearningForVisionAndMultimedia"
    • Run name: "classical_baselines"
    • Tags:     ["Baseline_ClassWeights", "Baseline_Oversampling"]
    • Tutte le metriche sono prefissate con il tag dell'esperimento:
        Baseline_ClassWeights/train_loss, ...
        Baseline_Oversampling/train_loss, ...
"""

import os
import torch
import numpy as np
import wandb

from config import (
    DATASET_DIR, METRICS_DIR,
    RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_EPOCHS, RESNET_LR,
    NUM_WORKERS, PIN_MEMORY, SEED,
)
from dataset.loader import setup_dataset, get_dataloaders, get_balanced_train_dataloader
from train import train_resnet
from eval import evaluate_on_test
from utils.seed import set_seed


# ─── Costanti esperimento ─────────────────────────────────────────────────────
WANDB_PROJECT = "gan-chest-xray-augmentation"
WANDB_ENTITY  = "MachineLearningForVisionAndMultimedia"
WANDB_RUN_NAME = "classical_baselines"
WANDB_TAGS     = ["Baseline_ClassWeights", "Baseline_Oversampling"]

TAG_CW  = "Baseline_ClassWeights"
TAG_OS  = "Baseline_Oversampling"


def compute_class_weights(train_dir: str, class_names: list[str]) -> torch.Tensor:
    """
    Calcola i pesi delle classi con la formula 'balanced' di sklearn:
        w_i = N_total / (n_classes * count_i)

    Garantisce che classi meno frequenti abbiano un peso maggiore,
    penalizzando di più gli errori su di esse durante il training.

    Args:
        train_dir:    cartella radice del training set (ImageFolder layout)
        class_names:  lista ordinata dei nomi delle classi
                      (stessa dell'attributo .classes di ImageFolder)

    Returns:
        Tensor 1D di float32 con i pesi per ogni classe.
    """
    # Conta le immagini per ogni classe (sottocartelle in train_dir)
    counts = []
    for cls in class_names:
        cls_dir = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(
                f"Cartella classe non trovata: {cls_dir}\n"
                f"Assicurarsi che train_dir contenga una sottocartella per ogni classe."
            )
        n = len([
            f for f in os.listdir(cls_dir)
            if os.path.isfile(os.path.join(cls_dir, f))
        ])
        counts.append(n)

    counts = np.array(counts, dtype=np.float64)
    n_total = counts.sum()
    n_classes = len(counts)

    # Formula balanced: uguale alla sklearn class_weight='balanced'
    weights = n_total / (n_classes * counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    print(f"\n  [Class Weights] Distribuzione classi: "
          f"{dict(zip(class_names, counts.astype(int)))}")
    print(f"  [Class Weights] Pesi calcolati:       "
          f"{dict(zip(class_names, [f'{w:.4f}' for w in weights.tolist()]))}")

    return weights_tensor


def run_baseline_class_weights(train_dir, val_dir, test_dir, classes, device):
    """
    Esperimento A: ResNet con CrossEntropyLoss pesata.

    1. Calcola i pesi delle classi inversamente proporzionali alla frequenza.
    2. Crea i dataloader standard (nessun sampler bilanciato).
    3. Allena train_resnet passando class_weights.
    4. Valuta sul test set.

    Returns:
        (model, history, ckpt_path, report_dict, cm)
    """
    print(f"\n{'='*60}")
    print(f"  ESPERIMENTO A — {TAG_CW}")
    print(f"{'='*60}")

    # Pesi per la loss
    cw = compute_class_weights(train_dir, classes)

    # DataLoader standard (shuffle, nessun sampler)
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE
    )

    # Training con loss pesata
    model, history, ckpt_path = train_resnet(
        train_loader, val_loader, device,
        epochs=RESNET_EPOCHS, lr=RESNET_LR,
        tag=TAG_CW,
        class_weights=cw
    )

    # Valutazione finale sul test set
    report_dict, cm = evaluate_on_test(
        model, ckpt_path, test_loader, classes, device,
        tag=TAG_CW, out_dir=METRICS_DIR
    )

    return model, history, ckpt_path, report_dict, cm


def run_baseline_oversampling(train_dir, val_dir, test_dir, classes, device):
    """
    Esperimento B: ResNet con WeightedRandomSampler sulle immagini reali.

    1. Crea il train_loader bilanciato via WeightedRandomSampler (dati reali).
    2. Crea val_loader e test_loader standard.
    3. Allena train_resnet con loss standard (nessun peso).
    4. Valuta sul test set.

    Returns:
        (model, history, ckpt_path, report_dict, cm)
    """
    print(f"\n{'='*60}")
    print(f"  ESPERIMENTO B — {TAG_OS}")
    print(f"{'='*60}")

    # Train loader bilanciato con WeightedRandomSampler (dati reali)
    train_loader, _ = get_balanced_train_dataloader(
        train_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE
    )

    # Val e Test loader standard (nessun sampler, shuffle=False)
    _, val_loader, test_loader, _ = get_dataloaders(
        train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE
    )

    # Training con loss standard (il bilanciamento è nel sampler)
    model, history, ckpt_path = train_resnet(
        train_loader, val_loader, device,
        epochs=RESNET_EPOCHS, lr=RESNET_LR,
        tag=TAG_OS,
        class_weights=None
    )

    # Valutazione finale sul test set
    report_dict, cm = evaluate_on_test(
        model, ckpt_path, test_loader, classes, device,
        tag=TAG_OS, out_dir=METRICS_DIR
    )

    return model, history, ckpt_path, report_dict, cm


def log_comparison_summary(report_cw: dict, report_os: dict, classes: list[str]):
    """
    Logga su WandB una tabella riassuntiva con le metriche chiave dei due
    esperimenti, per facilitare il confronto con la pipeline GAN.
    """
    print(f"\n{'='*60}")
    print("  RIEPILOGO CONFRONTO BASELINE")
    print(f"{'='*60}")

    header = f"  {'Metrica':<30} {'ClassWeights':>14} {'Oversampling':>14}"
    print(header)
    print(f"  {'-'*58}")

    metrics_to_log = {}

    for cls in classes:
        for m in ["precision", "recall", "f1-score"]:
            v_cw = report_cw[cls][m]
            v_os = report_os[cls][m]
            label = f"{cls} {m}"
            print(f"  {label:<30} {v_cw:>14.4f} {v_os:>14.4f}")
            metrics_to_log[f"Comparison/{cls}_{m}_CW"] = v_cw
            metrics_to_log[f"Comparison/{cls}_{m}_OS"] = v_os

    for key in ["accuracy"]:
        v_cw = report_cw[key]
        v_os = report_os[key]
        label = f"Overall {key}"
        print(f"  {label:<30} {v_cw:>14.4f} {v_os:>14.4f}")
        metrics_to_log[f"Comparison/overall_{key}_CW"] = v_cw
        metrics_to_log[f"Comparison/overall_{key}_OS"] = v_os

    for key in ["macro avg", "weighted avg"]:
        for m in ["precision", "recall", "f1-score"]:
            v_cw = report_cw[key][m]
            v_os = report_os[key][m]
            label = f"{key} {m}"
            print(f"  {label:<30} {v_cw:>14.4f} {v_os:>14.4f}")
            safe_key = key.replace(" ", "_")
            metrics_to_log[f"Comparison/{safe_key}_{m}_CW"] = v_cw
            metrics_to_log[f"Comparison/{safe_key}_{m}_OS"] = v_os

    wandb.log(metrics_to_log)
    print(f"\n  ✅ Tabella comparativa loggata su WandB.")


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  BASELINE CLASSICHE — Confronto con pipeline GAN")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # ── WandB init ────────────────────────────────────────────────────────────
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        tags=WANDB_TAGS,
        config={
            "seed":               SEED,
            "resnet_img_size":    RESNET_IMG_SIZE,
            "resnet_batch_size":  RESNET_BATCH_SIZE,
            "resnet_epochs":      RESNET_EPOCHS,
            "resnet_lr":          RESNET_LR,
            "num_workers":        NUM_WORKERS,
            "pin_memory":         PIN_MEMORY,
            "experiment":         "classical_baselines",
            "method_A":           "CrossEntropyLoss(weight=class_weights)",
            "method_B":           "WeightedRandomSampler (real images only)",
        }
    )

    # Assi X personalizzati per le due baseline (evita sovrapposizioni su WandB)
    wandb.define_metric(f"{TAG_CW}/epoch")
    wandb.define_metric(f"{TAG_CW}/*", step_metric=f"{TAG_CW}/epoch")
    wandb.define_metric(f"{TAG_OS}/epoch")
    wandb.define_metric(f"{TAG_OS}/*", step_metric=f"{TAG_OS}/epoch")

    # ── Setup dataset ─────────────────────────────────────────────────────────
    print(f"\n[1/4] Caricamento dataset da: {DATASET_DIR}")
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        print("❌ Dataset non trovato. Uscita.")
        wandb.finish(exit_code=1)
        return
    train_dir, val_dir, test_dir = res

    # Leggi i nomi delle classi dall'ImageFolder (ordine alfabetico: NORMAL, PNEUMONIA)
    from torchvision.datasets import ImageFolder
    _tmp_ds = ImageFolder(root=train_dir)
    classes = _tmp_ds.classes
    del _tmp_ds
    print(f"  Classi rilevate: {classes}")

    os.makedirs(METRICS_DIR, exist_ok=True)

    # ── Esperimento A: Class Weights ──────────────────────────────────────────
    print(f"\n[2/4] Avvio {TAG_CW}...")
    _, hist_cw, _, report_cw, cm_cw = run_baseline_class_weights(
        train_dir, val_dir, test_dir, classes, device
    )

    # ── Esperimento B: Oversampling ───────────────────────────────────────────
    print(f"\n[3/4] Avvio {TAG_OS}...")
    _, hist_os, _, report_os, cm_os = run_baseline_oversampling(
        train_dir, val_dir, test_dir, classes, device
    )

    # ── Riepilogo comparativo ─────────────────────────────────────────────────
    print(f"\n[4/4] Log riepilogo su WandB...")
    log_comparison_summary(report_cw, report_os, classes)

    print(f"\n{'='*60}")
    print(f"  ✅ Pipeline baseline completata!")
    print(f"  📊 Metriche salvate in: {METRICS_DIR}")
    print(f"  📊 WandB run: {WANDB_RUN_NAME} — project: {WANDB_PROJECT}")
    print(f"{'='*60}\n")

    wandb.finish()


if __name__ == "__main__":
    main()
