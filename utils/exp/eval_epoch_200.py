"""
eval_epoch_200.py
─────────────────
Script standalone che esegue la valutazione finale usando i pesi del Generator
salvati all'epoca 200 della WGAN-GP.

Pipeline:
  1. Carica il Generator + pesi epoca 200
  2. Genera immagini sintetiche per bilanciare la classe minoritaria (NORMAL)
  3. Crea augmented_dataset/train  (train originale + sintetiche)
  4. Allena ResNet sul dataset augmented e valida su val set
  5. Valutazione finale sul test set originale
  6. Plotta training curves, confusion matrix estesa e logga tutto su WandB

Tutto il codice di addestramento / generazione / valutazione è delegato ai
moduli già esistenti nel progetto; questo script si occupa solo di
orchestrare la pipeline.
"""

import os
import shutil
import torch
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

# ╔══════════════════════════════════════════════════════════════╗
# ║               ✏️  USER CONFIGURATION  ✏️                     ║
# ║  Modifica solo questa sezione per adattare lo script.       ║
# ╚══════════════════════════════════════════════════════════════╝

# ── Epoca del checkpoint Generator da caricare ───────────────────────────────
TARGET_EPOCH = 200          # ← cambia qui se vuoi valutare un'epoca diversa

# ── Path input: dataset originale ────────────────────────────────────────────
# Deve contenere le sottocartelle  train/ val/ test/  con NORMAL/ e PNEUMONIA/
DATASET_DIR = "./data/modified_dataset"

# ── Path input: cartella dei checkpoint GAN ───────────────────────────────────
# Lo script cercherà qui il file  G_epoch_{TARGET_EPOCH}.pth
GAN_CHECKPOINTS_DIR = "/content/drive/MyDrive/ProgettoMLVM/results_BAGAN/gan_checkpoints"

# ── Path output: cartella root dei risultati ─────────────────────────────────
RESULTS_DIR = "./results"

# ── Path output: dove salvare le immagini sintetiche generate ────────────────
SYNTHETIC_DIR = "./results/synthetic_images"

# ── Path output: dove costruire il dataset augmented ─────────────────────────
# Verrà creata la sottocartella train/ al suo interno
AUGMENTED_DIR = "./results/augmented_dataset"

# ── Path output: dove salvare report, plot e confusion matrix ────────────────
METRICS_DIR = "./results/metrics"

# ── ResNet: iperparametri di training ────────────────────────────────────────
RESNET_IMG_SIZE   = 128    # dimensione immagine (pixel)
RESNET_BATCH_SIZE = 32     # batch size
RESNET_EPOCHS     = 10     # epoche di training ResNet
RESNET_LR         = 0.001  # learning rate

# ── WandB: configurazione progetto ───────────────────────────────────────────
WANDB_PROJECT  = "gan-chest-xray-augmentation"
WANDB_ENTITY   = "MachineLearningForVisionAndMultimedia"
WANDB_RUN_NAME = f"eval_epoch_{TARGET_EPOCH}"   # ← rinomina la run se vuoi

# ╔══════════════════════════════════════════════════════════════╗
# ║            Fine USER CONFIGURATION                          ║
# ╚══════════════════════════════════════════════════════════════╝

# ─── Tag interno (derivato automaticamente) ──────────────────────────────────
EVAL_TAG = f"Phase3_ep{TARGET_EPOCH}"   # usato per report e metriche WandB

# ─── Costanti architettura GAN (non toccare) ─────────────────────────────────
# Devono corrispondere esattamente ai valori usati durante il training.
from config import GAN_NZ, GAN_N_CLASS, GAN_NC, GAN_D, SEED

# ─── Moduli progetto ─────────────────────────────────────────────────────────
from dataset.loader import setup_dataset, get_dataloaders
from models.wgan import Generator
from eval import generate_synthetic_images, evaluate_on_test
from train import train_resnet
from utils.seed import set_seed


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    """Stampa un banner visibile per separare le fasi."""
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def _count_class(folder: str, cls: str) -> int:
    """Conta i file in una sottocartella di classe (ignora file nascosti)."""
    path = os.path.join(folder, cls)
    if not os.path.isdir(path):
        return 0
    return sum(1 for f in os.listdir(path) if not f.startswith('.'))


def _plot_training_curves(history: dict, tag: str, out_dir: str) -> str:
    """
    Plotta le 4 curve di training (train loss, val loss, val acc, macro F1)
    su una figura 2×2, le salva su disco e le logga su WandB.
    Restituisce il path del file salvato.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Curves — {tag}", fontsize=15, fontweight='bold')

    metrics_cfg = [
        ("train_loss",   "Train Loss",       "Loss",     "#e63946"),
        ("val_loss",     "Val Loss",         "Loss",     "#457b9d"),
        ("val_acc",      "Val Accuracy (%)", "Accuracy", "#2a9d8f"),
        ("val_macro_f1", "Val Macro F1",     "F1",       "#e9c46a"),
    ]

    for ax, (key, title, ylabel, color) in zip(axes.flat, metrics_cfg):
        values = history.get(key, [])
        epochs = list(range(1, len(values) + 1))
        ax.plot(epochs, values, marker='o', color=color, linewidth=2, markersize=4)

        # Evidenzia il punto migliore
        if values:
            best_idx = (values.index(min(values)) if 'loss' in key
                        else values.index(max(values)))
            ax.scatter(epochs[best_idx], values[best_idx],
                       color='black', zorder=5, s=80,
                       label=f"Best: {values[best_idx]:.4f} (ep.{epochs[best_idx]})")
            ax.legend(fontsize=9)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"training_curves_{tag}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    wandb.log({
        f"{tag}/Training_Curves": wandb.Image(save_path,
                                              caption=f"Training curves — {tag}")
    })
    print(f"  📈 Curve di training salvate in: {save_path}")
    return save_path


def _plot_confusion_matrix_extended(cm, class_names: list, tag: str, out_dir: str) -> str:
    """
    Plotta la confusion matrix assoluta (counts) e quella normalizzata (%)
    affiancate in una figura 1×2, le salva su disco e le logga su WandB.
    Restituisce il path del file salvato.
    """
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix — {tag}", fontsize=14, fontweight='bold')

    # Assoluta
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title("Counts (absolute)")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")

    # Normalizzata %
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title("Normalized (%)")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"cm_extended_{tag}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    wandb.log({
        f"{tag}/Confusion_Matrix_Extended": wandb.Image(
            save_path, caption=f"Confusion matrix abs+norm — {tag}")
    })
    print(f"  🗂️  Confusion matrix estesa salvata in: {save_path}")
    return save_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Riproducibilità ──────────────────────────────────────────────────────
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _banner(f"eval_epoch_{TARGET_EPOCH}.py  —  device: {device}")

    # ── WandB ────────────────────────────────────────────────────────────────
    print("\n[1/7] Inizializzazione WandB …")
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        config={
            "target_epoch":        TARGET_EPOCH,
            "seed":                SEED,
            "resnet_img_size":     RESNET_IMG_SIZE,
            "resnet_batch_size":   RESNET_BATCH_SIZE,
            "resnet_epochs":       RESNET_EPOCHS,
            "resnet_lr":           RESNET_LR,
            "gan_nz":              GAN_NZ,
            "gan_n_class":         GAN_N_CLASS,
            "gan_checkpoints_dir": GAN_CHECKPOINTS_DIR,
            "dataset_dir":         DATASET_DIR,
        }
    )
    # Asse X personalizzato per le metriche epoch-level emesse da train_resnet
    wandb.define_metric(f"{EVAL_TAG}/epoch")
    wandb.define_metric(f"{EVAL_TAG}/*", step_metric=f"{EVAL_TAG}/epoch")
    print("  ✅ WandB inizializzato.")

    # ── STEP 1: Setup dataset ────────────────────────────────────────────────
    _banner("STEP 1 — Setup dataset originale")
    result = setup_dataset(dataset_dir=DATASET_DIR)
    if result is None:
        print("  ❌ Dataset non trovato. Interrompo.")
        wandb.finish()
        return
    train_dir, val_dir, test_dir = result

    n_normal    = _count_class(train_dir, 'NORMAL')
    n_pneumonia = _count_class(train_dir, 'PNEUMONIA')
    num_gen_normal    = max(0, n_pneumonia - n_normal)  # colma il gap NORMAL
    num_gen_pneumonia = 0                               # PNEUMONIA già maggioritaria

    print(f"\n  Train originale → NORMAL: {n_normal}  |  PNEUMONIA: {n_pneumonia}")
    print(f"  Immagini sintetiche da generare → NORMAL: {num_gen_normal}  |  PNEUMONIA: {num_gen_pneumonia}")

    # Log statistiche dataset originale
    wandb.log({
        "Dataset/train_normal_original":    n_normal,
        "Dataset/train_pneumonia_original": n_pneumonia,
        "Dataset/class_imbalance_ratio":    round(n_pneumonia / max(n_normal, 1), 3),
        "Dataset/num_gen_normal":           num_gen_normal,
        "Dataset/num_gen_pneumonia":        num_gen_pneumonia,
    })

    # ── STEP 2: Carica Generator epoca TARGET_EPOCH ───────────────────────────
    _banner(f"STEP 2 — Caricamento Generator (epoca {TARGET_EPOCH})")
    g_ckpt_path = os.path.join(GAN_CHECKPOINTS_DIR, f"G_epoch_{TARGET_EPOCH}.pth")

    if not os.path.isfile(g_ckpt_path):
        print(f"  ❌ Checkpoint non trovato: {g_ckpt_path}")
        print("     Assicurati che il training GAN abbia completato le epoche richieste.")
        wandb.finish()
        return

    G = Generator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=GAN_D).to(device)
    G.load_state_dict(torch.load(g_ckpt_path, map_location=device))
    G.eval()
    print(f"  ✅ Generator caricato da: {g_ckpt_path}")

    # ── STEP 3: Generazione immagini sintetiche ───────────────────────────────
    _banner("STEP 3 — Generazione immagini sintetiche")
    if num_gen_normal == 0 and num_gen_pneumonia == 0:
        print("  ℹ️  Il dataset è già bilanciato, nessuna generazione necessaria.")
    else:
        if os.path.exists(SYNTHETIC_DIR):
            shutil.rmtree(SYNTHETIC_DIR)

        generate_synthetic_images(
            G,
            num_gen_normal    = num_gen_normal,
            num_gen_pneumonia = num_gen_pneumonia,
            nz      = GAN_NZ,
            n_class = GAN_N_CLASS,
            device  = device,
            syn_dir = SYNTHETIC_DIR,
        )
        print(f"  ✅ Sintetiche salvate in: {SYNTHETIC_DIR}")

    # ── STEP 4: Costruzione augmented_dataset/train ───────────────────────────
    _banner("STEP 4 — Costruzione augmented_dataset/train")
    aug_train_dir = os.path.join(AUGMENTED_DIR, 'train')

    if os.path.exists(AUGMENTED_DIR):
        print(f"  🗑️  Rimozione augmented_dir precedente: {AUGMENTED_DIR}")
        shutil.rmtree(AUGMENTED_DIR)

    print(f"  📋 Copia train originale → {aug_train_dir} …")
    shutil.copytree(train_dir, aug_train_dir)

    for cls in ['NORMAL', 'PNEUMONIA']:
        syn_cls = os.path.join(SYNTHETIC_DIR, cls)
        aug_cls = os.path.join(aug_train_dir, cls)
        if os.path.isdir(syn_cls):
            for fname in os.listdir(syn_cls):
                shutil.copy(os.path.join(syn_cls, fname),
                            os.path.join(aug_cls, fname))

    n_aug_normal    = _count_class(aug_train_dir, 'NORMAL')
    n_aug_pneumonia = _count_class(aug_train_dir, 'PNEUMONIA')
    print(f"\n  Augmented train → NORMAL: {n_aug_normal}  |  PNEUMONIA: {n_aug_pneumonia}")
    print(f"  ✅ Dataset augmented creato in: {aug_train_dir}")

    # Log bilanciamento dataset augmented
    wandb.log({
        "Dataset/aug_train_normal":    n_aug_normal,
        "Dataset/aug_train_pneumonia": n_aug_pneumonia,
        "Dataset/aug_balance_ratio":   round(n_aug_pneumonia / max(n_aug_normal, 1), 3),
        "Dataset/synthetic_added":     num_gen_normal + num_gen_pneumonia,
    })

    # ── STEP 5: DataLoaders ───────────────────────────────────────────────────
    _banner("STEP 5 — Creazione DataLoaders")
    aug_train_loader, val_loader, test_loader, classes = get_dataloaders(
        aug_train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE
    )
    print(f"  Classi: {classes}")
    print(f"  Train batches: {len(aug_train_loader)}  |  "
          f"Val batches: {len(val_loader)}  |  "
          f"Test batches: {len(test_loader)}")
    print("  ✅ DataLoaders pronti.")

    # ── STEP 6: Training ResNet su dataset augmented ──────────────────────────
    _banner("STEP 6 — Training ResNet (dataset augmented)")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # train_resnet logga su WandB ad ogni epoca:
    #   {EVAL_TAG}/train_loss, val_loss, val_acc, val_macro_f1, epoch
    model_p3, hist_p3, ckpt_p3 = train_resnet(
        aug_train_loader, val_loader, device,
        epochs = RESNET_EPOCHS,
        lr     = RESNET_LR,
        tag    = EVAL_TAG,
    )
    print(f"\n  ✅ ResNet addestrata. Checkpoint migliore: {ckpt_p3}")

    # Plotta e logga le training curves
    _plot_training_curves(hist_p3, tag=EVAL_TAG, out_dir=METRICS_DIR)

    # ── STEP 7: Valutazione finale sul test set ───────────────────────────────
    _banner("STEP 7 — Valutazione finale sul test set")

    # evaluate_on_test logga su WandB:
    #   • {tag}/Confusion_Matrix_Image
    #   • {tag}_Test/Accuracy, Macro_F1, NORMAL_F1, PNEUMONIA_F1
    #   • {tag}_Test/ROC_Curve, PR_Curve
    report, cm = evaluate_on_test(
        model_p3, ckpt_p3, test_loader,
        class_names = classes,
        device      = device,
        tag         = EVAL_TAG,
        out_dir     = METRICS_DIR,
    )

    # Plotta la confusion matrix estesa (assoluta + normalizzata %)
    _plot_confusion_matrix_extended(cm, class_names=classes,
                                    tag=EVAL_TAG, out_dir=METRICS_DIR)

    # ── Riepilogo ─────────────────────────────────────────────────────────────
    _banner("RIEPILOGO FINALE")
    acc          = report.get("accuracy", 0.0)
    macro_f1     = report.get("macro avg",  {}).get("f1-score", 0.0)
    normal_f1    = report.get(classes[0],   {}).get("f1-score", 0.0) if classes else 0.0
    pneumonia_f1 = report.get(classes[1],   {}).get("f1-score", 0.0) if len(classes) > 1 else 0.0

    print(f"  🏁 Accuracy     : {acc * 100:.2f}%")
    print(f"  🏁 Macro F1     : {macro_f1:.4f}")
    print(f"  🏁 F1 NORMAL    : {normal_f1:.4f}")
    print(f"  🏁 F1 PNEUMONIA : {pneumonia_f1:.4f}")
    print(f"\n  📊 Report e plot salvati in: {METRICS_DIR}")
    print(f"  🔖 WandB run : {WANDB_PROJECT} / {WANDB_RUN_NAME}")

    # Valori di summary su WandB (visibili nella colonna della run nella dashboard)
    wandb.summary[f"{EVAL_TAG}/test_accuracy"]     = acc
    wandb.summary[f"{EVAL_TAG}/test_macro_f1"]     = macro_f1
    wandb.summary[f"{EVAL_TAG}/test_normal_f1"]    = normal_f1
    wandb.summary[f"{EVAL_TAG}/test_pneumonia_f1"] = pneumonia_f1
    wandb.summary["best_val_macro_f1"] = (max(hist_p3["val_macro_f1"])
                                          if hist_p3["val_macro_f1"] else 0.0)
    wandb.summary["best_val_accuracy"] = (max(hist_p3["val_acc"])
                                          if hist_p3["val_acc"] else 0.0)

    wandb.finish()
    print("\n  ✅ Pipeline completata con successo!\n")


if __name__ == "__main__":
    main()
