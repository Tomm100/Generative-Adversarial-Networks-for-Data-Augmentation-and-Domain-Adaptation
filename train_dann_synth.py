"""
Training loop per DANN Synthetic Domain Adaptation (Synth→Real).

Differenze rispetto al training DANN standard:
  - task loss (CrossEntropy) calcolata SOLO sulle immagini REALI (Source)
  - domain loss (BCE) calcolata su ENTRAMBI i domini (Source=0, Target=1)
  - il GRL forza il feature extractor a non distinguere reale da sintetico
  - λ cresce sigmoidalmente da 0 a 1 durante il training (Ganin et al., 2016)
  - model selection basata su Macro F1 sul val set REALE
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm import tqdm

from config import METRICS_DIR


def compute_lambda_p(p):
    """
    Scheduling dinamico di λ (Ganin et al., 2016).
    λ = 2 / (1 + exp(-10·p)) - 1,  p ∈ [0, 1]
    Cresce sigmoideamente da ~0 (nessun reversal) a ~1 (reversal pieno).
    """
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


def train_dann_synth(
    model, source_loader, target_loader, device,
    epochs=50,
    lr_feature=1e-4,
    lr_classifier=1e-3,
    beta1=0.5,
    tag="DANN_Synth",
    checkpoints_dir="./results/dann_synth_checkpoints",
    val_loader=None,
    class_names=None
):
    """
    Training loop DANN Synth→Real.

    Args:
        model:          DANNSynth instance
        source_loader:  DataLoader immagini REALI con label di classe
        target_loader:  DataLoader immagini SINTETICHE (label ignorate)
        device:         torch.device
        epochs:         numero epoche
        lr_feature:     LR per il Feature Extractor (pretrained — va preservato)
        lr_classifier:  LR per Label Predictor e Domain Discriminator
        beta1:          β₁ di Adam (0.5 per stabilità con GRL)
        tag:            identificativo per WandB e checkpoint
        checkpoints_dir: cartella checkpoint
        val_loader:     DataLoader val set REALE (per model selection)
        class_names:    ['NORMAL', 'PNEUMONIA']

    Returns:
        (model, history, best_ckpt_path)
    """
    os.makedirs(checkpoints_dir, exist_ok=True)
    model = model.to(device)

    # Optimizer con LR differenziati:
    # Feature extractor (pretrained) → LR basso per non distruggere i pesi
    # Classificatori (task + domain) → LR alto per convergenza rapida
    optimizer = optim.Adam([
        {'params': model.feature_extractor.parameters(),   'lr': lr_feature},
        {'params': model.label_predictor.parameters(),     'lr': lr_classifier},
        {'params': model.domain_discriminator.parameters(),'lr': lr_classifier},
    ], betas=(beta1, 0.999))

    criterion_class  = nn.CrossEntropyLoss()     # task: NORMAL vs PNEUMONIA
    criterion_domain = nn.BCEWithLogitsLoss()    # domain: reale (0) vs sintetico (1)

    ckpt_path = os.path.join(checkpoints_dir, f'best_{tag}.pth')
    best_val_f1 = -1.0
    best_weights = None

    history = {
        'loss_class': [], 'loss_domain': [], 'loss_total': [],
        'lambda_p': [], 'val_acc': [], 'val_macro_f1': []
    }

    n_batches = min(len(source_loader), len(target_loader))
    total_steps = epochs * n_batches

    print(f"\n{'='*60}")
    print(f"  DANN Synth→Real Training — {epochs} epoche")
    print(f"  Source (reali): {len(source_loader.dataset)} img")
    print(f"  Target (synth): {len(target_loader.dataset)} img")
    print(f"  LR Feature Extractor: {lr_feature}")
    print(f"  LR Classificatori:    {lr_classifier}")
    print(f"  Batch/dominio: {source_loader.batch_size}  |  n_batches/epoch: {n_batches}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss_class  = 0.0
        epoch_loss_domain = 0.0
        epoch_loss_total  = 0.0

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        pbar = tqdm(range(n_batches),
                    desc=f"  Epoch {epoch+1}/{epochs}", leave=False)

        for batch_idx in pbar:
            # Progresso globale e λ crescente
            p = (epoch * n_batches + batch_idx) / total_steps
            lambda_p = compute_lambda_p(p)

            # Sample dai due domini
            x_real, y_real = next(source_iter)           # reali con label
            x_fake, _      = next(target_iter)           # sintetici senza label

            x_real, y_real = x_real.to(device), y_real.to(device)
            x_fake         = x_fake.to(device)

            # Domain labels: 0 = reale (Source), 1 = sintetico (Target)
            d_real = torch.zeros(x_real.size(0), 1, device=device)
            d_fake = torch.ones(x_fake.size(0),  1, device=device)

            # Forward: Source (reale)
            class_out, domain_out_real, _ = model(x_real, lambda_p)

            # Forward: Target (sintetico) — solo per la domain loss
            _, domain_out_fake, _ = model(x_fake, lambda_p)

            # Task loss: classificazione SOLO sulle immagini REALI
            loss_class = criterion_class(class_out, y_real)

            # Domain loss: Source + Target (forza l'allineamento)
            loss_domain = (
                criterion_domain(domain_out_real, d_real) +
                criterion_domain(domain_out_fake, d_fake)
            )

            # Loss totale (il GRL inverte il gradiente domain → feature extractor)
            loss_total = loss_class + loss_domain

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            epoch_loss_class  += loss_class.item()
            epoch_loss_domain += loss_domain.item()
            epoch_loss_total  += loss_total.item()

            pbar.set_postfix({
                'cls': f"{loss_class.item():.3f}",
                'dom': f"{loss_domain.item():.3f}",
                'λ':   f"{lambda_p:.3f}"
            })

        # Medie epoca
        avg_cls = epoch_loss_class  / n_batches
        avg_dom = epoch_loss_domain / n_batches
        avg_tot = epoch_loss_total  / n_batches
        final_lambda = compute_lambda_p((epoch + 1) / epochs)

        history['loss_class'].append(avg_cls)
        history['loss_domain'].append(avg_dom)
        history['loss_total'].append(avg_tot)
        history['lambda_p'].append(final_lambda)

        # Valutazione su val set REALE
        val_acc, val_f1 = 0.0, 0.0
        if val_loader is not None:
            val_acc, val_f1 = _quick_eval(model, val_loader, device)

        history['val_acc'].append(val_acc)
        history['val_macro_f1'].append(val_f1)

        # WandB logging
        wandb.log({
            f"{tag}/loss_class":    avg_cls,
            f"{tag}/loss_domain":   avg_dom,
            f"{tag}/loss_total":    avg_tot,
            f"{tag}/lambda_p":      final_lambda,
            f"{tag}/val_acc":       val_acc,
            f"{tag}/val_macro_f1":  val_f1,
            f"{tag}/epoch":         epoch + 1,
        })

        # Best model selection su val Macro F1
        saved = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            saved = " ← best"

        elapsed = (time.time() - start_time) / 60
        eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
        print(f"  Epoch {epoch+1}/{epochs} | CLS: {avg_cls:.4f} | "
              f"DOM: {avg_dom:.4f} | λ: {final_lambda:.3f} | "
              f"ValAcc: {val_acc:.2f}% | ValF1: {val_f1:.4f}{saved} | "
              f"{elapsed:.1f}m (ETA: {eta:.1f}m)")

    # Salva best model
    if best_weights is not None:
        model.load_state_dict(best_weights)
        torch.save(best_weights, ckpt_path)
    else:
        best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        torch.save(best_weights, ckpt_path)
        print("  ⚠️  Nessun val loader: salvato stato finale.")

    print(f"\n  Best Val Macro F1: {best_val_f1:.4f}")
    print(f"  Checkpoint: {ckpt_path}")

    return model, history, ckpt_path


def _quick_eval(model, loader, device):
    """Valutazione rapida (Accuracy + Macro F1) con λ=0 (nessun reversal)."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            class_out, _, _ = model(x, lambda_=0.0)
            _, pred = torch.max(class_out, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())

    model.train()
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    acc = 100.0 * correct / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, f1


def evaluate_dann_synth(model, ckpt_path, test_loader, class_names, device,
                        tag="DANN_Synth", out_dir=None):
    """
    Valutazione completa sul test set REALE con i best weights.
    Salva classification report, confusion matrix e loga su WandB.
    """
    if out_dir is None:
        out_dir = METRICS_DIR
    os.makedirs(out_dir, exist_ok=True)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            class_out, _, _ = model(x, lambda_=0.0)
            _, pred = torch.max(class_out, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())

    print(f"\n{'='*60}")
    print(f"  RISULTATI {tag} — Test Set REALE")
    print(f"{'='*60}")

    report_dict = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True)
    report_str  = classification_report(
        all_labels, all_preds, target_names=class_names)
    print(report_str)

    # Salva report
    report_path = os.path.join(out_dir, f'report_{tag}.txt')
    with open(report_path, 'w') as f:
        f.write(f"RISULTATI {tag} — Test Set REALE\n{'='*60}\n{report_str}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — {tag}')
    plt.tight_layout()
    cm_path = os.path.join(out_dir, f'cm_{tag}.png')
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)

    wandb.log({
        f"{tag}/Confusion_Matrix":    wandb.Image(cm_path),
        f"{tag}_Test/Accuracy":       report_dict["accuracy"],
        f"{tag}_Test/Macro_F1":       report_dict["macro avg"]["f1-score"],
        f"{tag}_Test/NORMAL_F1":      report_dict[class_names[0]]["f1-score"],
        f"{tag}_Test/NORMAL_Recall":  report_dict[class_names[0]]["recall"],
        f"{tag}_Test/PNEUMONIA_F1":   report_dict[class_names[1]]["f1-score"],
    })

    return report_dict, cm
