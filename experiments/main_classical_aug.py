"""
Ablation Study — Data Augmentation Classica per Chest X-Ray.

Trasformazioni clinicamente appropriate (da paper):
  Rotazione leggera:  −5° ÷ +5°  (riscontrata nella pratica clinica)
  Traslazione:        ±5% H/W    (posizionamento variabile del paziente)
  Scaling uniforme:   0.95 – 1.05 (zoom leggero, uguale su x e y)

Trasformazioni escluse (clinicamente scorrette):
  Riflessione (x e y) — immagini non fisiologiche
  Rotazioni severe    — X-ray irrealistici
  Scaling asimmetrico — deformazione anatomica
  Shearing            — distorsioni inesistenti

Pipeline:
  1. Allena ResNet baseline sul dataset sbilanciato originale
  2. Per ogni percentuale di gap (25, 50, 75, 100):
     a. Genera copie augmentate della classe minoritaria (NORMAL)
     b. Allena ResNet sul dataset augmentato
     c. Valuta con evaluate_on_test (F1, Acc, CM)
  3. Stampa tabella riepilogativa e logga tutto su W&B

Uso:
  python experiments/main_classical_aug.py
"""

import os
import shutil
import random

import torch
import wandb
import numpy as np
from PIL import Image
from torchvision import transforms

# Root del repo nel path: consente l'esecuzione da experiments/ (python experiments/main_classical_aug.py)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATASET_DIR, RESULTS_DIR, METRICS_DIR,
    RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_NUM_CLASSES,
    RESNET_EPOCHS, RESNET_LR,
    SEED,
)
from dataset.loader import setup_dataset, get_dataloaders
from training.train import train_resnet
from evaluation.eval import evaluate_on_test
from utils.seed import set_seed


# CONFIGURAZIONE
OUT_DIR = os.path.join(RESULTS_DIR, "classical_aug")

# Percentuali di gap da testare nell'ablation study
ABLATION_PCTS = [25, 50, 75, 100]

# Trasformazioni clinicamente appropriate per chest X-ray
CLINICAL_AUGMENTATION = transforms.Compose([
    transforms.RandomRotation(degrees=5),                       # −5° ÷ +5°
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),                                 # ±5% shift
        scale=(0.95, 1.05),                                     # zoom uniforme ±5%
    ),
])


# GENERAZIONE IMMAGINI AUGMENTATE

def augment_minority_class(train_dir, augmentation_pct, out_dir):
    """
    Genera copie augmentate della classe minoritaria (NORMAL) per colmare
    il gap di sbilanciamento al `augmentation_pct`%.

    Returns:
        aug_train_dir (str): percorso della cartella di training augmentata
    """
    normal_dir = os.path.join(train_dir, 'NORMAL')
    pneumonia_dir = os.path.join(train_dir, 'PNEUMONIA')

    n_normal = len(os.listdir(normal_dir))
    n_pneumonia = len(os.listdir(pneumonia_dir))
    gap = n_pneumonia - n_normal
    num_to_generate = int(gap * (augmentation_pct / 100.0))

    print(f"\n  Dataset originale: {n_normal} NORMAL, {n_pneumonia} PNEUMONIA")
    print(f"  Gap totale:        {gap} immagini")
    print(f"  Gap da colmare:    {augmentation_pct}% -> {num_to_generate} immagini augmentate")

    # Copia il dataset originale nella directory di output
    aug_train_dir = os.path.join(out_dir, f"augmented_train_{augmentation_pct}pct")
    if os.path.exists(aug_train_dir):
        shutil.rmtree(aug_train_dir)
    shutil.copytree(train_dir, aug_train_dir)

    # Genera le immagini augmentate
    aug_normal_dir = os.path.join(aug_train_dir, 'NORMAL')
    source_files = [
        f for f in os.listdir(normal_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not source_files:
        print("  ERRORE: Nessuna immagine trovata nella cartella NORMAL.")
        return aug_train_dir

    print(f"  Generazione di {num_to_generate} immagini augmentate dalla classe NORMAL...")

    generated = 0
    while generated < num_to_generate:
        src_file = random.choice(source_files)
        src_path = os.path.join(normal_dir, src_file)

        img = Image.open(src_path).convert('RGB')
        aug_img = CLINICAL_AUGMENTATION(img)

        base, ext = os.path.splitext(src_file)
        aug_filename = f"aug_{generated:05d}_{base}{ext}"
        aug_img.save(os.path.join(aug_normal_dir, aug_filename))

        generated += 1

    n_aug_normal = len(os.listdir(aug_normal_dir))
    n_aug_pneumonia = len(os.listdir(os.path.join(aug_train_dir, 'PNEUMONIA')))
    print(f"\n  Dataset augmentato: {n_aug_normal} NORMAL + {n_aug_pneumonia} PNEUMONIA")

    return aug_train_dir


# MAIN — ABLATION STUDY

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ──
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        print("ERRORE: dataset non trovato.")
        return
    train_dir, val_dir, test_dir = res

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE
    )

    # ── W&B Init ──
    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        name="ClassicalAug_Ablation",
        config={
            "seed":              SEED,
            "resnet_img_size":   RESNET_IMG_SIZE,
            "resnet_batch_size": RESNET_BATCH_SIZE,
            "resnet_epochs":     RESNET_EPOCHS,
            "resnet_lr":         RESNET_LR,
            "ablation_pcts":     ABLATION_PCTS,
            "augmentation_type": "Classical (Rotation ±5°, Translation ±5%, Scale ±5%)",
        }
    )

    wandb.define_metric("Baseline/epoch")
    wandb.define_metric("Baseline/*", step_metric="Baseline/epoch")
    for pct in ABLATION_PCTS:
        wandb.define_metric(f"ClassicalAug_{pct}pct/epoch")
        wandb.define_metric(f"ClassicalAug_{pct}pct/*", step_metric=f"ClassicalAug_{pct}pct/epoch")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print("  STEP 1: Training Baseline ResNet (dataset sbilanciato)")
    print(f"{'='*60}")

    set_seed(SEED)
    model_baseline, hist_baseline, ckpt_baseline = train_resnet(
        train_loader, val_loader, device,
        epochs=RESNET_EPOCHS, lr=RESNET_LR, tag="Baseline"
    )

    report_baseline, cm_baseline = evaluate_on_test(
        model_baseline, ckpt_baseline, test_loader, class_names, device,
        tag="Baseline", out_dir=OUT_DIR
    )

    ablation_results = {
        "Baseline": report_baseline
    }

    for pct in ABLATION_PCTS:
        print(f"\n{'='*60}")
        print(f"  ABLATION — Classical Augmentation al {pct}% del gap")
        print(f"{'='*60}")
        print("  Trasformazioni applicate:")
        print("    • Rotazione:    ±5°")
        print("    • Traslazione:  ±5% (H e W)")
        print("    • Scaling:      ±5% (uniforme)")

        aug_train_dir = augment_minority_class(train_dir, pct, OUT_DIR)

        aug_train_loader, _, _, _ = get_dataloaders(
            aug_train_dir, val_dir, test_dir,
            img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE
        )

        tag = f"ClassicalAug_{pct}pct"

        set_seed(SEED)
        model_aug, hist_aug, ckpt_aug = train_resnet(
            aug_train_loader, val_loader, device,
            epochs=RESNET_EPOCHS, lr=RESNET_LR, tag=tag
        )

        report_aug, cm_aug = evaluate_on_test(
            model_aug, ckpt_aug, test_loader, class_names, device,
            tag=tag, out_dir=OUT_DIR
        )

        ablation_results[f"{pct}%"] = report_aug

        # Pulizia dataset augmentato per risparmiare spazio
        shutil.rmtree(aug_train_dir)
        print(f"  Rimossa cartella temporanea: {aug_train_dir}")

    print(f"\n{'='*70}")
    print("  RIEPILOGO ABLATION STUDY — CLASSICAL AUGMENTATION")
    print(f"{'='*70}")

    # Header
    col_labels = ["Baseline"] + [f"{p}%" for p in ABLATION_PCTS]
    header = f"  {'Metrica':<20}" + "".join(f"{c:>14}" for c in col_labels)
    print(header)
    print(f"  {'-' * (20 + 14 * len(col_labels))}")

    # Macro F1
    row = f"  {'Macro F1':<20}"
    base_f1 = ablation_results["Baseline"]["macro avg"]["f1-score"]
    row += f"{base_f1:>14.4f}"
    for pct in ABLATION_PCTS:
        f1 = ablation_results[f"{pct}%"]["macro avg"]["f1-score"]
        diff = f1 - base_f1
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
        row += f"  {f1:.4f}({arrow}{abs(diff):.3f})"
    print(row)

    # Accuracy
    row = f"  {'Accuracy':<20}"
    base_acc = ablation_results["Baseline"]["accuracy"]
    row += f"{base_acc:>14.4f}"
    for pct in ABLATION_PCTS:
        acc = ablation_results[f"{pct}%"]["accuracy"]
        diff = acc - base_acc
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
        row += f"  {acc:.4f}({arrow}{abs(diff):.3f})"
    print(row)

    # Per-class F1
    for cls in class_names:
        row = f"  {cls + ' F1':<20}"
        base_cls_f1 = ablation_results["Baseline"][cls]["f1-score"]
        row += f"{base_cls_f1:>14.4f}"
        for pct in ABLATION_PCTS:
            cls_f1 = ablation_results[f"{pct}%"][cls]["f1-score"]
            diff = cls_f1 - base_cls_f1
            arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
            row += f"  {cls_f1:.4f}({arrow}{abs(diff):.3f})"
        print(row)

    # ── W&B: tabella riepilogativa ──
    columns = ["Gap %", "Macro F1", "Accuracy"] + [f"{c} F1" for c in class_names]
    table_data = []

    # Riga baseline
    table_data.append([
        "Baseline",
        ablation_results["Baseline"]["macro avg"]["f1-score"],
        ablation_results["Baseline"]["accuracy"],
    ] + [ablation_results["Baseline"][c]["f1-score"] for c in class_names])

    # Righe ablation
    for pct in ABLATION_PCTS:
        r = ablation_results[f"{pct}%"]
        table_data.append([
            f"{pct}%",
            r["macro avg"]["f1-score"],
            r["accuracy"],
        ] + [r[c]["f1-score"] for c in class_names])

    wandb.log({
        "Ablation/Summary_Table": wandb.Table(columns=columns, data=table_data),
    })

    # Metriche scalari per confronto diretto
    for pct in ABLATION_PCTS:
        r = ablation_results[f"{pct}%"]
        wandb.log({
            f"Ablation/MacroF1_{pct}pct": r["macro avg"]["f1-score"],
            f"Ablation/Accuracy_{pct}pct": r["accuracy"],
        })

    wandb.finish()

    print(f"\n  Report e plot salvati in: {OUT_DIR}")
    print(f"  Checkpoint baseline:     {ckpt_baseline}")
    print("  Esperimento loggato su W&B ")
    print("\nAblation Study Classical Augmentation completato!")


if __name__ == "__main__":
    main()
