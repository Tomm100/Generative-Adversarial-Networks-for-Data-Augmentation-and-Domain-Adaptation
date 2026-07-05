"""
Analisi ROC/AUC e Precision-Recall per il confronto:
  1. Baseline  — ResNet allenata su dataset sbilanciato originale
  2. Finale    — ResNet su dataset augmentato con SNGAN 128 PG+BG (75% gap)

Pipeline:
  - Allena la ResNet baseline sul dataset sbilanciato
  - Carica il generatore SNGAN, genera immagini sintetiche al 75% del gap
  - Crea il dataset augmentato e allena una seconda ResNet
  - Calcola curve ROC e PR (per-classe e macro) per entrambi i modelli
  - Salva plot e logga tutto su Weights & Biases
"""

import os
import shutil
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import wandb

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    classification_report,
)

from config import (
    DATASET_DIR, RESULTS_DIR, METRICS_DIR,
    RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_NUM_CLASSES,
    RESNET_EPOCHS, RESNET_LR,
    GAN_NZ, GAN_N_CLASS, GAN_NC,
    SEED,
)
from dataset.loader import setup_dataset, get_dataloaders
from models.resnet import ResNetClassifier
from models.sngan_128 import SNGenerator as Generator   # SNGAN 128 PG+BG: generatore SNGAN
from train import train_resnet
from eval import generate_synthetic_images
from utils.seed import set_seed


# ==============================================================================
# PERCORSI DEI PESI — MODIFICA QUI
# ==============================================================================
# Pesi del Generatore SNGAN 128 PG+BG
SNGAN_GEN_CKPT = "/percorso/ai/pesi/G_epoch_XXX.pth"
# Dimensione del generatore (GAN_D)
GEN_D = 128
# Percentuale del gap da colmare (75%)
AUGMENTATION_PCT = 75
# ==============================================================================


OUT_DIR = os.path.join(RESULTS_DIR, "roc_pr_analysis")


# ==============================================================================
# FUNZIONI DI ANALISI
# ==============================================================================

def collect_predictions(model, test_loader, device):
    """Raccoglie le probabilità e le etichette dal test set."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_roc_curves(y_true, y_prob, class_names):
    """Calcola le curve ROC per ogni classe e la macro-average."""
    n_classes = len(class_names)
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        y_bin = (y_true == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(y_bin, y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


def compute_pr_curves(y_true, y_prob, class_names):
    """Calcola le curve Precision-Recall per ogni classe e la macro-average."""
    n_classes = len(class_names)
    precision, recall, avg_prec = {}, {}, {}

    for i in range(n_classes):
        y_bin = (y_true == i).astype(int)
        precision[i], recall[i], _ = precision_recall_curve(y_bin, y_prob[:, i])
        avg_prec[i] = average_precision_score(y_bin, y_prob[:, i])

    avg_prec["macro"] = np.mean([avg_prec[i] for i in range(n_classes)])

    return precision, recall, avg_prec


# ==============================================================================
# FUNZIONI DI PLOTTING
# ==============================================================================

def plot_roc_comparison(roc_baseline, roc_final, class_names, out_dir):
    """Genera plot ROC side-by-side: Baseline vs Finale."""
    fpr_b, tpr_b, auc_b = roc_baseline
    fpr_f, tpr_f, auc_f = roc_final
    n_classes = len(class_names)

    colors = ['#2196F3', '#FF5722', '#4CAF50']
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, fpr, tpr, roc_auc, title in [
        (axes[0], fpr_b, tpr_b, auc_b, "Baseline (Dataset Sbilanciato)"),
        (axes[1], fpr_f, tpr_f, auc_f, f"Finale (SNGAN 128 PG+BG — {AUGMENTATION_PCT}%)")
    ]:
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc[i]:.4f})')

        ax.plot(fpr["macro"], tpr["macro"], color=colors[2], lw=2.5, linestyle='--',
                label=f'Macro-avg (AUC = {roc_auc["macro"]:.4f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('ROC Curve — Confronto Baseline vs Finale', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'roc_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_pr_comparison(pr_baseline, pr_final, class_names, out_dir):
    """Genera plot PR side-by-side: Baseline vs Finale."""
    prec_b, rec_b, ap_b = pr_baseline
    prec_f, rec_f, ap_f = pr_final
    n_classes = len(class_names)

    colors = ['#2196F3', '#FF5722', '#4CAF50']
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, prec, rec, ap, title in [
        (axes[0], prec_b, rec_b, ap_b, "Baseline (Dataset Sbilanciato)"),
        (axes[1], prec_f, rec_f, ap_f, f"Finale (SNGAN 128 PG+BG — {AUGMENTATION_PCT}%)")
    ]:
        for i in range(n_classes):
            ax.plot(rec[i], prec[i], color=colors[i], lw=2,
                    label=f'{class_names[i]} (AP = {ap[i]:.4f})')

        ax.axhline(y=0.5, color='k', linestyle='--', lw=1, alpha=0.4)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Precision-Recall Curve — Confronto Baseline vs Finale', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'pr_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_roc_overlay(roc_baseline, roc_final, class_names, out_dir):
    """ROC overlay — stessa classe, due modelli sovrapposti."""
    fpr_b, tpr_b, auc_b = roc_baseline
    fpr_f, tpr_f, auc_f = roc_final
    n_classes = len(class_names)

    fig, axes = plt.subplots(1, n_classes + 1, figsize=(7 * (n_classes + 1), 6))

    for i in range(n_classes):
        ax = axes[i]
        ax.plot(fpr_b[i], tpr_b[i], color='#EF5350', lw=2,
                label=f'Baseline (AUC = {auc_b[i]:.4f})')
        ax.plot(fpr_f[i], tpr_f[i], color='#66BB6A', lw=2,
                label=f'Finale (AUC = {auc_f[i]:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
        ax.set_title(f'{class_names[i]}', fontsize=13, fontweight='bold')
        ax.set_xlabel('FPR', fontsize=11)
        ax.set_ylabel('TPR', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Macro
    ax = axes[n_classes]
    ax.plot(fpr_b["macro"], tpr_b["macro"], color='#EF5350', lw=2, linestyle='--',
            label=f'Baseline Macro (AUC = {auc_b["macro"]:.4f})')
    ax.plot(fpr_f["macro"], tpr_f["macro"], color='#66BB6A', lw=2, linestyle='--',
            label=f'Finale Macro (AUC = {auc_f["macro"]:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
    ax.set_title('Macro-Average', fontsize=13, fontweight='bold')
    ax.set_xlabel('FPR', fontsize=11)
    ax.set_ylabel('TPR', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('ROC Overlay — Baseline vs Finale (per classe)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'roc_overlay.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_pr_overlay(pr_baseline, pr_final, class_names, out_dir):
    """PR overlay — stessa classe, due modelli sovrapposti."""
    prec_b, rec_b, ap_b = pr_baseline
    prec_f, rec_f, ap_f = pr_final
    n_classes = len(class_names)

    fig, axes = plt.subplots(1, n_classes + 1, figsize=(7 * (n_classes + 1), 6))

    for i in range(n_classes):
        ax = axes[i]
        ax.plot(rec_b[i], prec_b[i], color='#EF5350', lw=2,
                label=f'Baseline (AP = {ap_b[i]:.4f})')
        ax.plot(rec_f[i], prec_f[i], color='#66BB6A', lw=2,
                label=f'Finale (AP = {ap_f[i]:.4f})')
        ax.set_title(f'{class_names[i]}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Macro AP (text annotation)
    ax = axes[n_classes]
    ax.text(0.5, 0.6, f'Baseline Macro AP: {ap_b["macro"]:.4f}',
            ha='center', va='center', fontsize=14, color='#EF5350', fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.5, 0.4, f'Finale Macro AP: {ap_f["macro"]:.4f}',
            ha='center', va='center', fontsize=14, color='#66BB6A', fontweight='bold',
            transform=ax.transAxes)
    diff = ap_f["macro"] - ap_b["macro"]
    arrow = "↑" if diff > 0 else "↓"
    ax.text(0.5, 0.2, f'Δ = {arrow} {abs(diff):.4f}',
            ha='center', va='center', fontsize=13, fontweight='bold',
            transform=ax.transAxes)
    ax.set_title('Macro Average Precision', fontsize=13, fontweight='bold')
    ax.axis('off')

    plt.suptitle('Precision-Recall Overlay — Baseline vs Finale (per classe)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'pr_overlay.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def build_summary_table(roc_b, roc_f, pr_b, pr_f, class_names):
    """Costruisce una tabella wandb di riepilogo."""
    _, _, auc_b = roc_b
    _, _, auc_f = roc_f
    _, _, ap_b  = pr_b
    _, _, ap_f  = pr_f

    columns = ["Classe", "AUC Baseline", "AUC Finale", "Δ AUC",
               "AP Baseline", "AP Finale", "Δ AP"]
    data = []

    for i, name in enumerate(class_names):
        d_auc = auc_f[i] - auc_b[i]
        d_ap  = ap_f[i] - ap_b[i]
        data.append([name, auc_b[i], auc_f[i], d_auc,
                     ap_b[i], ap_f[i], d_ap])

    d_auc_macro = auc_f["macro"] - auc_b["macro"]
    d_ap_macro  = ap_f["macro"] - ap_b["macro"]
    data.append(["MACRO", auc_b["macro"], auc_f["macro"], d_auc_macro,
                 ap_b["macro"], ap_f["macro"], d_ap_macro])

    return wandb.Table(columns=columns, data=data)


def evaluate_model(model, test_loader, class_names, device, tag):
    """Esegue l'inferenza e calcola ROC + PR su un modello già caricato."""
    print(f"\n  [{tag}] Valutazione sul test set...")

    y_prob, y_true = collect_predictions(model, test_loader, device)
    y_pred = np.argmax(y_prob, axis=1)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=class_names)
    print(f"\n{report_str}")

    # ROC e PR
    roc = compute_roc_curves(y_true, y_prob, class_names)
    pr  = compute_pr_curves(y_true, y_prob, class_names)

    return y_prob, y_true, roc, pr, report


# ==============================================================================
# PIPELINE DI AUGMENTATION
# ==============================================================================

def create_augmented_dataset(train_dir, sngan_ckpt, device, augmentation_pct=75):
    """
    Carica il generatore SNGAN, genera immagini sintetiche per colmare
    il gap al `augmentation_pct`% e crea il dataset augmentato.

    Returns:
        aug_train_dir (str): percorso della cartella di training augmentata
    """
    # Calcolo gap
    n_train_n = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    n_train_p = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    max_deficit = n_train_p - n_train_n
    num_to_generate = int(max_deficit * (augmentation_pct / 100.0))

    print(f"\n  Dataset originale: {n_train_n} NORMAL, {n_train_p} PNEUMONIA")
    print(f"  Gap totale:        {max_deficit} immagini")
    print(f"  Gap da colmare:    {augmentation_pct}% → {num_to_generate} immagini sintetiche")

    # Caricamento generatore
    print(f"\n  Caricamento Generatore SNGAN da: {sngan_ckpt}")
    G = Generator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=GEN_D).to(device)
    G.load_state_dict(torch.load(sngan_ckpt, map_location=device))
    G.eval()

    # Generazione pool sintetico
    pool_dir = os.path.join(RESULTS_DIR, "roc_pr_synthetic_pool")
    if os.path.exists(pool_dir):
        shutil.rmtree(pool_dir)

    print("  Generazione immagini sintetiche...")
    generate_synthetic_images(
        G, num_gen_normal=num_to_generate, num_gen_pneumonia=0,
        nz=GAN_NZ, n_class=GAN_N_CLASS, device=device, syn_dir=pool_dir
    )

    # Creazione dataset augmentato
    aug_train_dir = os.path.join(RESULTS_DIR, "roc_pr_augmented_train")
    if os.path.exists(aug_train_dir):
        shutil.rmtree(aug_train_dir)
    shutil.copytree(train_dir, aug_train_dir)

    # Copia immagini sintetiche nella cartella NORMAL
    syn_normal_dir = os.path.join(pool_dir, 'NORMAL')
    if os.path.exists(syn_normal_dir):
        dest_normal_dir = os.path.join(aug_train_dir, 'NORMAL')
        syn_files = os.listdir(syn_normal_dir)
        random.shuffle(syn_files)
        for f in syn_files[:num_to_generate]:
            shutil.copy(
                os.path.join(syn_normal_dir, f),
                os.path.join(dest_normal_dir, f)
            )

    n_aug_n = len(os.listdir(os.path.join(aug_train_dir, 'NORMAL')))
    n_aug_p = len(os.listdir(os.path.join(aug_train_dir, 'PNEUMONIA')))
    print(f"\n  Dataset augmentato: {n_aug_n} NORMAL + {n_aug_p} PNEUMONIA")

    return aug_train_dir


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Verifica percorso pesi SNGAN ──
    if not os.path.exists(SNGAN_GEN_CKPT):
        print(f"ERRORE: Checkpoint SNGAN non trovato: {SNGAN_GEN_CKPT}")
        print("Modifica il percorso SNGAN_GEN_CKPT in cima al file.")
        return

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
        name=f"ROC_PR_Baseline_vs_SNGAN_{AUGMENTATION_PCT}pct",
        config={
            "seed":                SEED,
            "resnet_img_size":     RESNET_IMG_SIZE,
            "resnet_batch_size":   RESNET_BATCH_SIZE,
            "resnet_epochs":       RESNET_EPOCHS,
            "resnet_lr":           RESNET_LR,
            "sngan_ckpt":          SNGAN_GEN_CKPT,
            "augmentation_pct":    AUGMENTATION_PCT,
            "final_description":   f"SNGAN 128 PG+BG — {AUGMENTATION_PCT}% gap colmato",
            "analysis_type":       "ROC/AUC + Precision-Recall",
        }
    )

    wandb.define_metric("Baseline/epoch")
    wandb.define_metric("Baseline/*",    step_metric="Baseline/epoch")
    wandb.define_metric("Finale/epoch")
    wandb.define_metric("Finale/*",      step_metric="Finale/epoch")

    os.makedirs(OUT_DIR, exist_ok=True)

    # ==========================================================================
    # STEP 1: Training Baseline ResNet (dataset sbilanciato originale)
    # ==========================================================================
    print(f"\n{'='*60}")
    print("  STEP 1: Training Baseline ResNet (dataset sbilanciato)")
    print(f"{'='*60}")

    set_seed(SEED)
    model_baseline, hist_baseline, ckpt_baseline = train_resnet(
        train_loader, val_loader, device,
        epochs=RESNET_EPOCHS, lr=RESNET_LR, tag="Baseline"
    )

    # ==========================================================================
    # STEP 2: Augmentation con SNGAN e Training ResNet Finale
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"  STEP 2: Augmentation SNGAN 128 PG+BG al {AUGMENTATION_PCT}%")
    print(f"{'='*60}")

    aug_train_dir = create_augmented_dataset(
        train_dir, SNGAN_GEN_CKPT, device, augmentation_pct=AUGMENTATION_PCT
    )

    # Creiamo i dataloader per il dataset augmentato
    aug_train_loader, _, _, _ = get_dataloaders(
        aug_train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE
    )

    print(f"\n{'='*60}")
    print(f"  STEP 3: Training ResNet Finale (dataset augmentato {AUGMENTATION_PCT}%)")
    print(f"{'='*60}")

    set_seed(SEED)
    model_finale, hist_finale, ckpt_finale = train_resnet(
        aug_train_loader, val_loader, device,
        epochs=RESNET_EPOCHS, lr=RESNET_LR, tag="Finale"
    )

    # ==========================================================================
    # STEP 4: Analisi ROC/PR
    # ==========================================================================
    print(f"\n{'='*60}")
    print("  STEP 4: Analisi ROC/AUC e Precision-Recall")
    print(f"{'='*60}")

    # Valutazione Baseline
    print(f"\n{'='*60}")
    print("  VALUTAZIONE BASELINE")
    print(f"{'='*60}")
    y_prob_b, y_true_b, roc_b, pr_b, report_b = evaluate_model(
        model_baseline, test_loader, class_names, device, "Baseline")

    # Valutazione Finale
    print(f"\n{'='*60}")
    print(f"  VALUTAZIONE FINALE (SNGAN 128 PG+BG — {AUGMENTATION_PCT}%)")
    print(f"{'='*60}")
    y_prob_f, y_true_f, roc_f, pr_f, report_f = evaluate_model(
        model_finale, test_loader, class_names, device, "Finale")

    # ── Plot ──
    print("\nGenerazione plot...")
    roc_comp_path = plot_roc_comparison(roc_b, roc_f, class_names, OUT_DIR)
    pr_comp_path  = plot_pr_comparison(pr_b, pr_f, class_names, OUT_DIR)
    roc_over_path = plot_roc_overlay(roc_b, roc_f, class_names, OUT_DIR)
    pr_over_path  = plot_pr_overlay(pr_b, pr_f, class_names, OUT_DIR)

    # ── Stampa riepilogo ──
    _, _, auc_b = roc_b
    _, _, auc_f = roc_f
    _, _, ap_b = pr_b
    _, _, ap_f = pr_f

    print(f"\n{'='*60}")
    print("  RIEPILOGO AUC / AP")
    print(f"{'='*60}")
    header = f"  {'Classe':<16} {'AUC Base':>10} {'AUC Fin':>10} {'Δ AUC':>10}  |  {'AP Base':>10} {'AP Fin':>10} {'Δ AP':>10}"
    print(header)
    print("  " + "-" * 90)

    for i, name in enumerate(class_names):
        d_auc = auc_f[i] - auc_b[i]
        d_ap  = ap_f[i] - ap_b[i]
        arrow_auc = "↑" if d_auc > 0 else "↓" if d_auc < 0 else "="
        arrow_ap  = "↑" if d_ap > 0 else "↓" if d_ap < 0 else "="
        print(f"  {name:<16} {auc_b[i]:>10.4f} {auc_f[i]:>10.4f} {arrow_auc}{abs(d_auc):>9.4f}  |  "
              f"{ap_b[i]:>10.4f} {ap_f[i]:>10.4f} {arrow_ap}{abs(d_ap):>9.4f}")

    d_auc_m = auc_f["macro"] - auc_b["macro"]
    d_ap_m  = ap_f["macro"] - ap_b["macro"]
    arrow_auc_m = "↑" if d_auc_m > 0 else "↓"
    arrow_ap_m  = "↑" if d_ap_m > 0 else "↓"
    print("  " + "-" * 90)
    print(f"  {'MACRO':<16} {auc_b['macro']:>10.4f} {auc_f['macro']:>10.4f} {arrow_auc_m}{abs(d_auc_m):>9.4f}  |  "
          f"{ap_b['macro']:>10.4f} {ap_f['macro']:>10.4f} {arrow_ap_m}{abs(d_ap_m):>9.4f}")

    # ── Log su W&B ──
    summary_table = build_summary_table(roc_b, roc_f, pr_b, pr_f, class_names)

    wandb.log({
        # Immagini
        "ROC_PR/ROC_Comparison":  wandb.Image(roc_comp_path),
        "ROC_PR/PR_Comparison":   wandb.Image(pr_comp_path),
        "ROC_PR/ROC_Overlay":     wandb.Image(roc_over_path),
        "ROC_PR/PR_Overlay":      wandb.Image(pr_over_path),
        # Tabella riepilogativa
        "ROC_PR/Summary_Table":   summary_table,
        # Metriche scalari
        "ROC_PR/Baseline_Macro_AUC":  auc_b["macro"],
        "ROC_PR/Finale_Macro_AUC":    auc_f["macro"],
        "ROC_PR/Delta_Macro_AUC":     d_auc_m,
        "ROC_PR/Baseline_Macro_AP":   ap_b["macro"],
        "ROC_PR/Finale_Macro_AP":     ap_f["macro"],
        "ROC_PR/Delta_Macro_AP":      d_ap_m,
    })

    # Log metriche per-classe
    for i, name in enumerate(class_names):
        wandb.log({
            f"ROC_PR/Baseline_AUC_{name}": auc_b[i],
            f"ROC_PR/Finale_AUC_{name}":   auc_f[i],
            f"ROC_PR/Baseline_AP_{name}":  ap_b[i],
            f"ROC_PR/Finale_AP_{name}":    ap_f[i],
        })

    # Curve interattive wandb (built-in)
    wandb.log({
        "ROC_PR/Baseline_ROC_Interactive": wandb.plot.roc_curve(
            y_true_b, y_prob_b.tolist(), labels=class_names, title="ROC Baseline"),
        "ROC_PR/Baseline_PR_Interactive": wandb.plot.pr_curve(
            y_true_b, y_prob_b.tolist(), labels=class_names, title="PR Baseline"),
        "ROC_PR/Finale_ROC_Interactive": wandb.plot.roc_curve(
            y_true_f, y_prob_f.tolist(), labels=class_names, title="ROC Finale"),
        "ROC_PR/Finale_PR_Interactive": wandb.plot.pr_curve(
            y_true_f, y_prob_f.tolist(), labels=class_names, title="PR Finale"),
    })

    # Salva report testuale
    report_path = os.path.join(OUT_DIR, "roc_pr_summary.txt")
    with open(report_path, "w") as f:
        f.write("ANALISI ROC/AUC e PRECISION-RECALL\n")
        f.write(f"Baseline vs SNGAN 128 PG+BG ({AUGMENTATION_PCT}% gap)\n")
        f.write("=" * 60 + "\n\n")
        for i, name in enumerate(class_names):
            f.write(f"{name}:\n")
            f.write(f"  AUC: {auc_b[i]:.4f} -> {auc_f[i]:.4f} (Δ {auc_f[i]-auc_b[i]:+.4f})\n")
            f.write(f"  AP:  {ap_b[i]:.4f} -> {ap_f[i]:.4f} (Δ {ap_f[i]-ap_b[i]:+.4f})\n\n")
        f.write(f"MACRO AUC: {auc_b['macro']:.4f} -> {auc_f['macro']:.4f} (Δ {d_auc_m:+.4f})\n")
        f.write(f"MACRO AP:  {ap_b['macro']:.4f} -> {ap_f['macro']:.4f} (Δ {d_ap_m:+.4f})\n")

    wandb.save(report_path)

    wandb.finish()

    print(f"\n  Plot salvati in:  {OUT_DIR}")
    print(f"  Report salvato:   {report_path}")
    print("  Esperimento loggato su W&B ✓")
    print("\nAnalisi ROC/PR completata!")


if __name__ == "__main__":
    main()
