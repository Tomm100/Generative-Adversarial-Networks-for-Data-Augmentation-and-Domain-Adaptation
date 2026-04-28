"""
Training e valutazione per CDAN (Conditional Adversarial Domain Adaptation).

Contiene:
  - compute_lambda_p():        scheduling dinamico di λ (identico a DANN)
  - compute_entropy_weights(): pesi per Entropy Conditioning
  - train_cdan():              loop di training CDAN con Entropy Conditioning
  - _quick_eval_cdan():        valutazione rapida durante il training
  - evaluate_cdan_on_target(): valutazione completa sul Target test set

Differenze rispetto a train_dann.py:
  1. Il forward del modello restituisce anche softmax_probs (4 valori invece di 3).
  2. Entropy Conditioning: ogni campione è pesato in base all'entropia delle
     predizioni. I campioni con bassa entropia (alta confidenza) contribuiscono
     di più alla domain loss, seguendo la formula:
       w = 1 + exp(-H(ŷ))  con  H(ŷ) = -Σ p_i · log(p_i + ε)
  3. La domain loss è calcolata campione per campione (reduction='none') e poi
     moltiplicata per i pesi prima della media, permettendo il weighting per-sample.

Riferimento:
  Long et al., "Conditional Adversarial Domain Adaptation", NeurIPS 2018.
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
# Lambda Scheduling (invariato rispetto a DANN)
# ═══════════════════════════════════════════════════════════════

def compute_lambda_p(p: float) -> float:
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
# Entropy Conditioning
# ═══════════════════════════════════════════════════════════════

def compute_entropy_weights(softmax_probs: torch.Tensor,
                            epsilon: float = 1e-8) -> torch.Tensor:
    """
    Calcola i pesi per l'Entropy Conditioning (Long et al., 2018).

    L'idea è che i campioni su cui il classificatore è già molto sicuro
    (bassa entropia) forniscono un segnale più affidabile per il discriminatore
    di dominio → peso maggiore. I campioni ambigui (alta entropia) vengono
    attenuati per non inquinare l'allineamento.

    Formula:
        H(ŷ) = -Σ_c  p_c · log(p_c + ε)           # entropia di Shannon
        w     = 1 + exp(-H(ŷ))                      # peso ∈ (1, 2]

    Proprietà:
        - H → 0  (certezza massima)  ⟹  w → 1 + exp(0)  = 2.0  (peso alto)
        - H → log(C) (max entropia)  ⟹  w → 1 + exp(-log C) ≈ 1.0 (peso basso)

    Args:
        softmax_probs: (B, num_classes) — probabilità softmax del classificatore.
                       Devono essere già dopo softmax (valori in [0, 1]).
        epsilon:       piccolo valore per stabilità numerica nel log.

    Returns:
        weights: (B,) — pesi per campione, detached dal computation graph
                 (non vogliamo gradienti attraverso i pesi stessi).
    """
    # Entropia di Shannon: (B,)
    entropy = -torch.sum(
        softmax_probs * torch.log(softmax_probs + epsilon),
        dim=1
    )
    # Peso: (B,)
    weights = 1.0 + torch.exp(-entropy)
    # Detach: i pesi sono usati come coefficienti scalari, non come nodi
    # del grafo computazionale (evita gradienti di secondo ordine indesiderati).
    return weights.detach()


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════

def train_cdan(model, source_loader, target_loader, device,
               epochs: int = 50,
               lr_feature: float = 1e-4,
               lr_classifier: float = 1e-3,
               beta1: float = 0.5,
               tag: str = "CDAN",
               checkpoints_dir=None,
               target_val_loader=None,
               class_names=None):
    """
    Training loop CDAN con:
      - Dual-domain sampling bilanciato (50/50 Source/Target per batch)
      - Prodotto multilineare (Conditional Coupling) per il discriminatore
      - Entropy Conditioning: pesi per-campione sulla domain loss
      - GRL con scheduling dinamico di λ
      - LR differenziati: basso per Feature Extractor, alto per classificatori
      - Validazione periodica su Target val set per model selection (Macro F1)

    Args:
        model:             CDAN_Model instance
        source_loader:     DataLoader Source domain (con label di classe)
        target_loader:     DataLoader Target domain (label ignorate — UDA)
        device:            torch.device
        epochs:            numero di epoche
        lr_feature:        learning rate per il Feature Extractor (pretrained)
        lr_classifier:     learning rate per Label Predictor e Domain Discriminator
        beta1:             β₁ di Adam (ridotto a 0.5 per stabilità con GRL)
        tag:               tag per logging WandB e naming checkpoint
        checkpoints_dir:   directory per salvare i checkpoint
        target_val_loader: DataLoader per validazione sul Target (model selection)
        class_names:       nomi delle classi per logging

    Returns:
        (model, history, best_ckpt_path)
          - model:          modello con i pesi del best checkpoint caricati
          - history:        dict con le curve di training
          - best_ckpt_path: path del checkpoint salvato
    """
    if checkpoints_dir is None:
        checkpoints_dir = DANN_CHECKPOINTS_DIR  # fallback sui checkpoints DANN

    model = model.to(device)

    # ── Optimizer Adam con LR differenziati ──
    # Il Feature Extractor usa LR più basso per non distruggere i pesi Phase 3.
    optimizer = optim.Adam([
        {'params': model.feature_extractor.parameters(),   'lr': lr_feature},
        {'params': model.label_predictor.parameters(),     'lr': lr_classifier},
        {'params': model.domain_discriminator.parameters(),'lr': lr_classifier},
    ], betas=(beta1, 0.999))

    # ── Loss Functions ──
    criterion_class = nn.CrossEntropyLoss()
    # reduction='none' → loss per-campione, necessaria per Entropy Weighting
    criterion_domain = nn.BCEWithLogitsLoss(reduction='none')

    # ── Setup checkpoint e tracking ──
    os.makedirs(checkpoints_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoints_dir, f'best_cdan_{tag}.pth')
    best_target_f1 = -1.0
    best_weights = None

    history = {
        'loss_class': [], 'loss_domain': [], 'loss_total': [],
        'lambda_p': [],
        'avg_entropy_source': [], 'avg_entropy_target': [],
        'target_acc': [], 'target_macro_f1': [],
    }

    n_batches = min(len(source_loader), len(target_loader))
    total_steps = epochs * n_batches

    print(f"\n{'='*60}")
    print(f"  CDAN Training — {epochs} epochs, {n_batches} batches/epoch")
    print(f"  Entropy Conditioning: ATTIVO")
    print(f"  LR Feature Extractor: {lr_feature}")
    print(f"  LR Classifiers:       {lr_classifier}")
    print(f"  Adam β₁:              {beta1}")
    print(f"  Total steps:          {total_steps}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss_class  = 0.0
        epoch_loss_domain = 0.0
        epoch_loss_total  = 0.0
        epoch_entropy_s   = 0.0
        epoch_entropy_t   = 0.0

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
            x_t, _   = next(target_iter)   # Label Target NON usate (UDA)

            x_s, y_s = x_s.to(device), y_s.to(device)
            x_t       = x_t.to(device)

            B_s = x_s.size(0)
            B_t = x_t.size(0)

            # Domain labels: 0 = Source, 1 = Target  →  (B, 1)
            domain_label_s = torch.zeros(B_s, 1, device=device)
            domain_label_t = torch.ones(B_t,  1, device=device)

            # ── Forward pass Source ──
            # Restituisce: (class_logits, domain_logits, features, softmax_probs)
            class_out_s, domain_out_s, _, probs_s = model(x_s, lambda_p)

            # ── Forward pass Target ──
            _, domain_out_t, _, probs_t = model(x_t, lambda_p)

            # ── Task Loss (solo Source) ──
            loss_class = criterion_class(class_out_s, y_s)

            # ── Entropy Weights (Entropy Conditioning) ──
            # Source: w_s ∈ (1, 2],  (B_s,)
            w_s = compute_entropy_weights(probs_s)
            # Target: w_t ∈ (1, 2],  (B_t,)
            w_t = compute_entropy_weights(probs_t)

            # ── Domain Loss pesata per-campione ──
            # criterion_domain restituisce (B, 1) grazie a reduction='none'
            # Squeeze → (B,), poi moltiplica per i pesi → media pesata
            dom_loss_s = criterion_domain(domain_out_s, domain_label_s).squeeze(1)  # (B_s,)
            dom_loss_t = criterion_domain(domain_out_t, domain_label_t).squeeze(1)  # (B_t,)

            loss_domain = (
                (w_s * dom_loss_s).mean() +
                (w_t * dom_loss_t).mean()
            )

            # ── Total Loss ──
            loss_total = loss_class + loss_domain

            # ── Backward + step ──
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # ── Accumula per logging ──
            epoch_loss_class  += loss_class.item()
            epoch_loss_domain += loss_domain.item()
            epoch_loss_total  += loss_total.item()

            # Entropia media per monitoring (detached, solo stat)
            with torch.no_grad():
                H_s = -torch.sum(probs_s * torch.log(probs_s + 1e-8), dim=1).mean().item()
                H_t = -torch.sum(probs_t * torch.log(probs_t + 1e-8), dim=1).mean().item()
            epoch_entropy_s += H_s
            epoch_entropy_t += H_t

            pbar.set_postfix({
                'cls': f"{loss_class.item():.3f}",
                'dom': f"{loss_domain.item():.3f}",
                'λ':   f"{lambda_p:.3f}",
                'H_t': f"{H_t:.3f}",
            })

        # ── Medie epoca ──
        avg_cls     = epoch_loss_class  / n_batches
        avg_dom     = epoch_loss_domain / n_batches
        avg_tot     = epoch_loss_total  / n_batches
        avg_H_s     = epoch_entropy_s   / n_batches
        avg_H_t     = epoch_entropy_t   / n_batches
        final_lambda = compute_lambda_p((epoch + 1) / epochs)

        history['loss_class'].append(avg_cls)
        history['loss_domain'].append(avg_dom)
        history['loss_total'].append(avg_tot)
        history['lambda_p'].append(final_lambda)
        history['avg_entropy_source'].append(avg_H_s)
        history['avg_entropy_target'].append(avg_H_t)

        # ── Valutazione su Target val set ──
        target_acc, target_f1 = 0.0, 0.0
        if target_val_loader is not None:
            target_acc, target_f1 = _quick_eval_cdan(model, target_val_loader, device)

        history['target_acc'].append(target_acc)
        history['target_macro_f1'].append(target_f1)

        # ── WandB Logging ──
        log_dict = {
            f"{tag}/loss_class":          avg_cls,
            f"{tag}/loss_domain":         avg_dom,
            f"{tag}/loss_total":          avg_tot,
            f"{tag}/lambda_p":            final_lambda,
            f"{tag}/entropy_source_avg":  avg_H_s,
            f"{tag}/entropy_target_avg":  avg_H_t,
            f"{tag}/epoch":               epoch + 1,
        }
        if target_val_loader is not None:
            log_dict[f"{tag}/target_val_acc"]      = target_acc
            log_dict[f"{tag}/target_val_macro_f1"] = target_f1
        wandb.log(log_dict)

        # ── Best model selection (su Target val Macro F1) ──
        saved = ""
        if target_f1 > best_target_f1:
            best_target_f1 = target_f1
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            saved = " ← best"

        elapsed = (time.time() - start_time) / 60
        eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)

        print(f"  Epoch {epoch+1}/{epochs} | CLS: {avg_cls:.4f} | "
              f"DOM: {avg_dom:.4f} | λ: {final_lambda:.3f} | "
              f"H_t: {avg_H_t:.3f} | "
              f"T-Acc: {target_acc:.2f}% | T-F1: {target_f1:.4f}{saved} | "
              f"{elapsed:.1f}m (ETA: {eta:.1f}m)")

    # ── Salva best model ──
    if best_weights is not None:
        model.load_state_dict(best_weights)
        torch.save(best_weights, ckpt_path)
    else:
        # Fallback: nessuna validazione disponibile → salva stato finale
        best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        torch.save(best_weights, ckpt_path)
        print(f"\n  ⚠️  Nessuna val loader: salvato stato finale come checkpoint.")

    print(f"\n  Best Target Val Macro F1: {best_target_f1:.4f}")
    print(f"  Checkpoint salvato in:    {ckpt_path}")

    return model, history, ckpt_path


# ═══════════════════════════════════════════════════════════════
# Evaluation Utilities
# ═══════════════════════════════════════════════════════════════

def _quick_eval_cdan(model, loader, device):
    """
    Valutazione rapida: Accuracy + Macro F1 sul loader fornito.
    Usata durante il training per model selection.
    λ=0 durante la valutazione (nessun gradient reversal).

    Gestisce il forward CDAN (4 valori di ritorno) in modo sicuro.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # CDAN forward: (class_logits, domain_logits, features, probs)
            class_out, _, _, _ = model(x, lambda_=0.0)
            _, pred = torch.max(class_out, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    model.train()  # Ripristina training mode

    correct  = sum(p == l for p, l in zip(all_preds, all_labels))
    acc      = 100.0 * correct / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, macro_f1


def evaluate_cdan_on_target(model, ckpt_path, test_loader, class_names, device,
                             tag="CDAN", out_dir=None):
    """
    Valutazione completa sul Target test set con il best checkpoint CDAN:
      - Accuracy, Macro F1-Score
      - Confusion Matrix (plot + salvataggio)
      - Classification Report completo
      - Log su WandB (metriche scalari + immagine CM)

    Interfaccia speculare a `evaluate_dann_on_target` per permettere
    il riuso di `log_dann_comparison` in main_cdan.py.

    Args:
        model:       CDAN_Model instance
        ckpt_path:   path del checkpoint (.pth) da caricare
        test_loader: DataLoader del Target test set
        class_names: lista nomi classi (es. ['NORMAL', 'PNEUMONIA'])
        device:      torch.device
        tag:         tag per logging WandB e naming file
        out_dir:     directory output per report e plot

    Returns:
        (report_dict, cm)
    """
    if out_dir is None:
        out_dir = METRICS_DIR
    os.makedirs(out_dir, exist_ok=True)

    # ── Carica best weights ──
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            class_out, _, _, _ = model(x, lambda_=0.0)
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
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
        f"{tag}/Confusion_Matrix":      wandb.Image(cm_path),
        f"{tag}_Test/Accuracy":         report_dict["accuracy"],
        f"{tag}_Test/Macro_F1":         report_dict["macro avg"]["f1-score"],
        f"{tag}_Test/NORMAL_F1":        report_dict[class_names[0]]["f1-score"],
        f"{tag}_Test/PNEUMONIA_F1":     report_dict[class_names[1]]["f1-score"],
    })

    print(f"\n  Report salvato in:          {report_path}")
    print(f"  Confusion Matrix salvata in: {cm_path}")

    return report_dict, cm
