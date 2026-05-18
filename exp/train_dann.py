"""
Training e valutazione per DANN (Domain-Adversarial Neural Network).

Contiene:
  - compute_lambda_p(): scheduling dinamico di λ
  - train_dann(): loop di training con dual-domain sampling
  - evaluate_dann_on_target(): valutazione completa sul Target test set
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import wandb
import matplotlib
matplotlib.use('Agg')  # Backend non interattivo per server/Colab
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)
from tqdm import tqdm

from config import DANN_CHECKPOINTS_DIR, METRICS_DIR


# ═══════════════════════════════════════════════════════════════
# Lambda Scheduling
# ═══════════════════════════════════════════════════════════════

def compute_lambda_p(p):
    """
    Scheduling dinamico di λ (Ganin et al., 2016).

        λ_p = 2 / (1 + exp(-10 · p)) - 1

    Args:
        p: progresso del training ∈ [0, 1]
           p = (epoch * n_batches + batch_idx) / (total_epochs * n_batches)

    Returns:
        λ ∈ [0, 1) — cresce sigmoideamente da 0 a ~1
    """
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════

def train_dann(model, source_loader, target_loader, device,
               epochs=50,
               lr_feature=1e-4, lr_classifier=1e-3,
               beta1=0.5,
               tag="DANN",
               checkpoints_dir=None,
               target_val_loader=None,
               class_names=None):
    """
    Training loop DANN con:
      - Dual-domain sampling bilanciato (50/50 Source/Target per batch)
      - Gradient Reversal Layer con scheduling dinamico di λ
      - LR differenziati: basso per Feature Extractor, alto per classificatori
      - Validazione periodica su Target val set per model selection

    Args:
        model: DANN_Model instance
        source_loader: DataLoader Source domain (con label di classe)
        target_loader: DataLoader Target domain (label ignorate)
        device: torch.device
        epochs: numero di epoche
        lr_feature: learning rate per il Feature Extractor (pretrained, va preservato)
        lr_classifier: learning rate per Label Predictor e Domain Discriminator
        beta1: β₁ di Adam (ridotto a 0.5 per stabilità con GRL)
        tag: tag per logging WandB
        checkpoints_dir: directory per salvare i checkpoint
        target_val_loader: DataLoader per validazione sul Target (model selection)
        class_names: nomi delle classi per logging

    Returns:
        (model, history, best_ckpt_path)
    """
    if checkpoints_dir is None:
        checkpoints_dir = DANN_CHECKPOINTS_DIR

    model = model.to(device)

    # ── Optimizer Adam con LR differenziati ──
    # Il Feature Extractor (source-pretrained) usa un LR 10× più basso
    # per non distruggere le feature già apprese sul Source domain.
    optimizer = optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': lr_feature},
        {'params': model.label_predictor.parameters(),   'lr': lr_classifier},
        {'params': model.domain_discriminator.parameters(), 'lr': lr_classifier},
    ], betas=(beta1, 0.999))

    # ── Loss Functions ──
    criterion_class = nn.CrossEntropyLoss()       # Classificazione (solo Source)
    criterion_domain = nn.BCEWithLogitsLoss()      # Dominio (Source + Target)

    # ── Tracking ──
    os.makedirs(checkpoints_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoints_dir, f'best_dann_{tag}.pth')
    best_target_f1 = -1.0
    best_weights = None

    history = {
        'loss_class': [], 'loss_domain': [], 'loss_total': [],
        'lambda_p': [],
        'target_acc': [], 'target_macro_f1': []
    }

    # Numero di batch per epoca = min dei due loader
    # (entrambi dovrebbero avere la stessa lunghezza grazie all'allineamento
    #  dei num_samples, ma usiamo min per sicurezza)
    n_batches = min(len(source_loader), len(target_loader))
    total_steps = epochs * n_batches

    print(f"\n{'='*60}")
    print(f"  DANN Training — {epochs} epochs, {n_batches} batches/epoch")
    print(f"  LR Feature Extractor: {lr_feature}")
    print(f"  LR Classifiers:       {lr_classifier}")
    print(f"  Adam β₁:              {beta1}")
    print(f"  Total steps:          {total_steps}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss_class = 0.0
        epoch_loss_domain = 0.0
        epoch_loss_total = 0.0

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        pbar = tqdm(range(n_batches),
                    desc=f"  Epoch {epoch+1}/{epochs}", leave=False)

        for batch_idx in pbar:
            # ── Calcolo progresso e λ ──
            p = (epoch * n_batches + batch_idx) / total_steps
            lambda_p = compute_lambda_p(p)

            # ── Sample da entrambi i domini ──
            x_s, y_s = next(source_iter)
            x_t, _ = next(target_iter)  # Label Target NON utilizzate (UDA)

            x_s, y_s = x_s.to(device), y_s.to(device)
            x_t = x_t.to(device)

            # Domain labels: 0 = Source, 1 = Target
            domain_label_s = torch.zeros(x_s.size(0), 1, device=device)
            domain_label_t = torch.ones(x_t.size(0), 1, device=device)

            # ── Forward pass ──
            # Source: usiamo sia class output che domain output
            class_out_s, domain_out_s, _ = model(x_s, lambda_p)
            # Target: usiamo solo domain output (no label supervisione)
            _, domain_out_t, _ = model(x_t, lambda_p)

            # ── Loss computation ──
            # Task loss: classificazione solo su Source
            loss_class = criterion_class(class_out_s, y_s)

            # Domain loss: discriminazione su entrambi i domini
            loss_domain = (
                criterion_domain(domain_out_s, domain_label_s) +
                criterion_domain(domain_out_t, domain_label_t)
            )

            # Total loss (il GRL gestisce l'inversione del gradiente internamente)
            loss_total = loss_class + loss_domain

            # ── Backward + step ──
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # ── Accumula loss ──
            epoch_loss_class += loss_class.item()
            epoch_loss_domain += loss_domain.item()
            epoch_loss_total += loss_total.item()

            pbar.set_postfix({
                'cls': f"{loss_class.item():.3f}",
                'dom': f"{loss_domain.item():.3f}",
                'λ': f"{lambda_p:.3f}"
            })

        # ── Medie epoca ──
        avg_cls = epoch_loss_class / n_batches
        avg_dom = epoch_loss_domain / n_batches
        avg_tot = epoch_loss_total / n_batches
        final_lambda = compute_lambda_p((epoch + 1) / epochs)

        history['loss_class'].append(avg_cls)
        history['loss_domain'].append(avg_dom)
        history['loss_total'].append(avg_tot)
        history['lambda_p'].append(final_lambda)

        # ── Valutazione su Target val set ──
        target_acc, target_f1 = 0.0, 0.0
        if target_val_loader is not None:
            target_acc, target_f1 = _quick_eval(model, target_val_loader, device)

        history['target_acc'].append(target_acc)
        history['target_macro_f1'].append(target_f1)

        # ── WandB Logging ──
        log_dict = {
            f"{tag}/loss_class": avg_cls,
            f"{tag}/loss_domain": avg_dom,
            f"{tag}/loss_total": avg_tot,
            f"{tag}/lambda_p": final_lambda,
            f"{tag}/epoch": epoch + 1,
        }
        if target_val_loader is not None:
            log_dict[f"{tag}/target_val_acc"] = target_acc
            log_dict[f"{tag}/target_val_macro_f1"] = target_f1
        wandb.log(log_dict)

        # ── Best model selection (su Target val F1) ──
        saved = ""
        if target_f1 > best_target_f1:
            best_target_f1 = target_f1
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            saved = " ← best"

        elapsed = (time.time() - start_time) / 60
        eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)

        print(f"  Epoch {epoch+1}/{epochs} | CLS: {avg_cls:.4f} | "
              f"DOM: {avg_dom:.4f} | λ: {final_lambda:.3f} | "
              f"T-Acc: {target_acc:.2f}% | T-F1: {target_f1:.4f}{saved} | "
              f"{elapsed:.1f}m (ETA: {eta:.1f}m)")

    # ── Salva best model ──
    if best_weights is not None:
        model.load_state_dict(best_weights)
        torch.save(best_weights, ckpt_path)
    else:
        # Fallback: nessuna validazione disponibile → salva lo stato finale
        best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        torch.save(best_weights, ckpt_path)
        print(f"\n  ⚠️  Nessuna val loader: salvato stato finale come checkpoint.")
    print(f"\n  Best Target Val Macro F1: {best_target_f1:.4f}")
    print(f"  Checkpoint salvato in: {ckpt_path}")

    return model, history, ckpt_path


# ═══════════════════════════════════════════════════════════════
# Evaluation Utilities
# ═══════════════════════════════════════════════════════════════

def _quick_eval(model, loader, device):
    """
    Valutazione rapida: Accuracy + Macro F1.
    Usata durante il training per model selection.
    λ=0 durante la valutazione (nessun gradient reversal).
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            class_out, _, _ = model(x, lambda_=0.0)
            _, pred = torch.max(class_out, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    model.train()  # Torna in training mode

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    acc = 100.0 * correct / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, macro_f1


def evaluate_dann_on_target(model, ckpt_path, test_loader, class_names, device,
                            tag="DANN", out_dir=None):
    """
    Valutazione completa sul Target test set:
      - Accuracy, Macro F1-Score
      - Confusion Matrix (plot + salvataggio)
      - Classification Report completo

    Args:
        model: DANN_Model instance
        ckpt_path: path del checkpoint da caricare
        test_loader: DataLoader del Target test set
        class_names: lista nomi classi (es. ['NORMAL', 'PNEUMONIA'])
        device: torch.device
        tag: tag per logging WandB e naming file
        out_dir: directory output per report e plot

    Returns:
        (report_dict, confusion_matrix)
    """
    if out_dir is None:
        out_dir = METRICS_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Carica best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            class_out, _, _ = model(x, lambda_=0.0)
            _, pred = torch.max(class_out, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())

    # ── Classification Report ──
    print(f"\n{'='*60}")
    print(f"  RISULTATI {tag} — Target Domain (Test Set)")
    print(f"{'='*60}")

    report_dict = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True)
    report_str = classification_report(
        all_labels, all_preds, target_names=class_names)
    print(report_str)

    # Salva report testuale
    report_path = os.path.join(out_dir, f'report_{tag}.txt')
    with open(report_path, 'w') as f:
        f.write(f"RISULTATI {tag} — Target Domain (Test Set)\n")
        f.write(f"{'='*60}\n")
        f.write(report_str)

    # ── Confusion Matrix ──
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — {tag} (Target Domain)')
    plt.tight_layout()

    cm_path = os.path.join(out_dir, f'cm_{tag}.png')
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)

    # ── WandB Logging ──
    wandb.log({
        f"{tag}/Confusion_Matrix": wandb.Image(cm_path),
        f"{tag}_Test/Accuracy": report_dict["accuracy"],
        f"{tag}_Test/Macro_F1": report_dict["macro avg"]["f1-score"],
        f"{tag}_Test/NORMAL_F1": report_dict[class_names[0]]["f1-score"],
        f"{tag}_Test/PNEUMONIA_F1": report_dict[class_names[1]]["f1-score"],
    })

    print(f"\n  Report salvato in: {report_path}")
    print(f"  Confusion Matrix salvata in: {cm_path}")

    return report_dict, cm
