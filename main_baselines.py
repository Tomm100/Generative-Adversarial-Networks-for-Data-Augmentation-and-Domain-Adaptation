"""
main_baselines.py
=================
Esegue i due esperimenti baseline per il bilanciamento del dataset senza GAN:

  Baseline A — Class Weighting:
      CrossEntropyLoss con pesi inversi alla frequenza di classe.
      Il DataLoader rimane invariato (shuffle=True, distribuzione originale sbilanciata).

  Baseline B — Oversampling:
      DataLoader con WeightedRandomSampler: NORMAL e PNEUMONIA sono equiprobabili
      in ogni batch. CrossEntropyLoss standard (nessun peso).

Entrambi gli esperimenti vengono loggati su un singolo run W&B con tag distinti
("ClassWeight" e "Oversampling"), in modo da poterli confrontare nella stessa
dashboard accanto a Phase1 (baseline puro) e Phase3 (GAN augmented).

Utilizzo:
    python main_baselines.py
"""

import torch
import os
import numpy as np
import wandb

from config import (
    DATASET_DIR, METRICS_DIR, RESULTS_DIR,
    RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_EPOCHS, RESNET_LR,
    NUM_WORKERS, PIN_MEMORY,
    SEED,
)
from dataset.loader import setup_dataset, get_dataloaders, get_balanced_train_dataloader
from train import train_resnet
from eval import evaluate_on_test
from utils.seed import set_seed


# ─── Helper: calcola class weights dal train_dir ─────────────────────────────

def compute_class_weights(train_dir: str, device: torch.device) -> torch.Tensor:
    """
    Calcola i pesi di classe come N_totale / (N_classi * N_per_classe).
    Questo schema è equivalente a sklearn compute_class_weight('balanced', ...).

    Returns:
        Tensor di forma [2] ordinato come ImageFolder:
        indice 0 → NORMAL, indice 1 → PNEUMONIA
    """
    from torchvision import datasets
    from torchvision.transforms import ToTensor

    # Carichiamo il dataset solo per contare le etichette (nessuna trasformazione costosa)
    ds = datasets.ImageFolder(root=train_dir, transform=ToTensor())
    labels = np.array([label for _, label in ds.samples])
    class_counts = np.bincount(labels)           # [n_normal, n_pneumonia]
    n_total = len(labels)
    n_classes = len(class_counts)

    # Bilanciamento "balanced": peso_i = N / (K * n_i)
    weights = n_total / (n_classes * class_counts.astype(float))
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    class_names = ds.classes
    for i, (name, cnt, w) in enumerate(zip(class_names, class_counts, weights)):
        print(f"  [{i}] {name}: {cnt} campioni → peso {w:.4f}")

    return weights_tensor.to(device)


# ─── Esperimento A: Class Weighting ──────────────────────────────────────────

def run_class_weight_baseline(train_dir, val_dir, test_dir, device):
    """
    Allena ResNet con CrossEntropyLoss pesata.
    DataLoader standard (sbilanciato), nessun oversampling.
    """
    print(f"\n{'='*60}")
    print("BASELINE A — CLASS WEIGHTING")
    print(f"{'='*60}")

    # DataLoader standard (stesso usato in Phase1)
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE
    )

    # Calcola i pesi di classe prima del training
    print("\n  [ClassWeight] Distribuzione training:")
    class_weights = compute_class_weights(train_dir, device)

    # Training con loss pesata
    model, hist, ckpt_path = train_resnet(
        train_loader, val_loader, device,
        epochs=RESNET_EPOCHS,
        lr=RESNET_LR,
        tag="ClassWeight",
        class_weights=class_weights
    )

    # Valutazione sul test set
    report, cm = evaluate_on_test(
        model, ckpt_path, test_loader, classes, device,
        tag="ClassWeight", out_dir=METRICS_DIR
    )

    return report, cm, hist, classes


# ─── Esperimento B: Oversampling ─────────────────────────────────────────────

def run_oversampling_baseline(train_dir, val_dir, test_dir, device):
    """
    Allena ResNet con WeightedRandomSampler (oversampling della classe NORMAL).
    Loss standard (nessun peso), val/test loader invariati.
    """
    print(f"\n{'='*60}")
    print("BASELINE B — OVERSAMPLING (WeightedRandomSampler)")
    print(f"{'='*60}")

    # DataLoader bilanciato per il training
    print("\n  [Oversampling] Distribuzione training:")
    train_loader, class_names = get_balanced_train_dataloader(
        train_dir,
        img_size=RESNET_IMG_SIZE,
        batch_size=RESNET_BATCH_SIZE
    )

    # Val e test loader standard (senza sampler: valutazione sulla distribuzione reale)
    _, val_loader, test_loader, classes = get_dataloaders(
        train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE
    )

    # Training con loss standard
    model, hist, ckpt_path = train_resnet(
        train_loader, val_loader, device,
        epochs=RESNET_EPOCHS,
        lr=RESNET_LR,
        tag="Oversampling",
        class_weights=None   # Nessun peso: il bilanciamento è nel sampler
    )

    # Valutazione sul test set
    report, cm = evaluate_on_test(
        model, ckpt_path, test_loader, classes, device,
        tag="Oversampling", out_dir=METRICS_DIR
    )

    return report, cm, hist, classes


# ─── Confronto finale ─────────────────────────────────────────────────────────

def print_comparison(report_cw, report_os, classes):
    """
    Stampa una tabella comparativa Macro F1 / per-classe tra le due baseline.
    """
    print(f"\n{'='*60}")
    print("CONFRONTO FINALE: ClassWeight vs Oversampling")
    print(f"{'='*60}")
    header = f"  {'Metrica':<28} {'ClassWeight':>12} {'Oversampling':>14}"
    print(header)
    print("  " + "-" * 56)

    for cls in classes:
        for metric in ['precision', 'recall', 'f1-score']:
            cw_val = report_cw[cls][metric]
            os_val = report_os[cls][metric]
            diff = os_val - cw_val
            arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
            label = f"{cls} {metric}"
            print(f"  {label:<28} {cw_val:>12.4f} {os_val:>14.4f}  {arrow} {abs(diff):.4f}")

    print("  " + "-" * 56)
    for metric_key, label in [
        ("macro avg", "Macro avg F1"),
        ("weighted avg", "Weighted avg F1"),
    ]:
        cw_val = report_cw[metric_key]["f1-score"]
        os_val = report_os[metric_key]["f1-score"]
        diff = os_val - cw_val
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
        print(f"  {label:<28} {cw_val:>12.4f} {os_val:>14.4f}  {arrow} {abs(diff):.4f}")

    cw_acc = report_cw["accuracy"]
    os_acc = report_os["accuracy"]
    diff = os_acc - cw_acc
    arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
    print(f"  {'Accuracy':<28} {cw_acc:>12.4f} {os_acc:>14.4f}  {arrow} {abs(diff):.4f}")

    # Logga su W&B il confronto sintetico
    wandb.log({
        "Baselines_Comparison/ClassWeight_MacroF1":   report_cw["macro avg"]["f1-score"],
        "Baselines_Comparison/Oversampling_MacroF1":  report_os["macro avg"]["f1-score"],
        "Baselines_Comparison/ClassWeight_Accuracy":  report_cw["accuracy"],
        "Baselines_Comparison/Oversampling_Accuracy": report_os["accuracy"],
    })


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- W&B init ---
    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        name="baselines_cw_vs_oversampling",
        config={
            "seed":              SEED,
            "resnet_img_size":   RESNET_IMG_SIZE,
            "resnet_batch_size": RESNET_BATCH_SIZE,
            "resnet_epochs":     RESNET_EPOCHS,
            "resnet_lr":         RESNET_LR,
            "experiments":       ["ClassWeight", "Oversampling"],
        }
    )

    # Assi X personalizzati per disaccoppiare i grafici delle due baseline
    wandb.define_metric("ClassWeight/epoch")
    wandb.define_metric("ClassWeight/*",    step_metric="ClassWeight/epoch")
    wandb.define_metric("Oversampling/epoch")
    wandb.define_metric("Oversampling/*",   step_metric="Oversampling/epoch")

    # --- Setup dataset ---
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    res = setup_dataset(dataset_dir=DATASET_DIR)
    if res is None:
        print("ERRORE: dataset non trovato. Controlla DATASET_DIR in config.py.")
        wandb.finish()
        return
    train_dir, val_dir, test_dir = res

    # --- Baseline A: Class Weighting ---
    report_cw, cm_cw, hist_cw, classes = run_class_weight_baseline(
        train_dir, val_dir, test_dir, device
    )

    # --- Baseline B: Oversampling ---
    report_os, cm_os, hist_os, _ = run_oversampling_baseline(
        train_dir, val_dir, test_dir, device
    )

    # --- Confronto finale ---
    print_comparison(report_cw, report_os, classes)

    print(f"\n📊 Report salvati in: {METRICS_DIR}")
    print("✅ Pipeline baselines completata.")

    wandb.finish()


if __name__ == "__main__":
    main()
