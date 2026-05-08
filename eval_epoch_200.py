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
  6. Stampa il report di classificazione e salva la confusion matrix

Tutto il codice di addestramento / generazione / valutazione è delegato ai
moduli già esistenti nel progetto; questo script si occupa solo di
orchestrare la pipeline.
"""

import os
import shutil
import torch
import wandb

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
GAN_CHECKPOINTS_DIR = "./Drive/MyDrive/ProgettoMLVM/results_BAGAN/gan_checkpoints"

# ── Path output: cartella root dei risultati ─────────────────────────────────
RESULTS_DIR = "./results"

# ── Path output: dove salvare le immagini sintetiche generate ────────────────
SYNTHETIC_DIR = "./results/synthetic_images"

# ── Path output: dove costruire il dataset augmented ─────────────────────────
# Verrà creata la sottocartella train/ al suo interno
AUGMENTED_DIR = "./results/augmented_dataset"

# ── Path output: dove salvare report e confusion matrix ──────────────────────
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Riproducibilità ──────────────────────────────────────────────────────
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _banner(f"eval_epoch_{TARGET_EPOCH}.py  —  device: {device}")

    # ── WandB ────────────────────────────────────────────────────────────────
    print("\n[1/6] Inizializzazione WandB …")
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        config={
            "target_epoch":      TARGET_EPOCH,
            "seed":              SEED,
            "resnet_img_size":   RESNET_IMG_SIZE,
            "resnet_batch_size": RESNET_BATCH_SIZE,
            "resnet_epochs":     RESNET_EPOCHS,
            "resnet_lr":         RESNET_LR,
            "gan_nz":            GAN_NZ,
            "gan_n_class":       GAN_N_CLASS,
        }
    )
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

    # ── STEP 2: Carica Generator epoca 200 ───────────────────────────────────
    _banner(f"STEP 2 — Caricamento Generator (epoca {TARGET_EPOCH})")
    g_ckpt_path = os.path.join(GAN_CHECKPOINTS_DIR, f"G_epoch_{TARGET_EPOCH}.pth")

    if not os.path.isfile(g_ckpt_path):
        print(f"  ❌ Checkpoint non trovato: {g_ckpt_path}")
        print("     Assicurati che il training GAN abbia completato le 200 epoche.")
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
        # Pulisce eventuali sintetiche precedenti per evitare contaminazione
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

    # Ricostruisce sempre da zero per garanzia di coerenza
    if os.path.exists(AUGMENTED_DIR):
        print(f"  🗑️  Rimozione augmented_dir precedente: {AUGMENTED_DIR}")
        shutil.rmtree(AUGMENTED_DIR)

    print(f"  📋 Copia train originale → {aug_train_dir} …")
    shutil.copytree(train_dir, aug_train_dir)

    # Copia le sintetiche nelle rispettive sottocartelle di classe
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

    model_p3, hist_p3, ckpt_p3 = train_resnet(
        aug_train_loader, val_loader, device,
        epochs = RESNET_EPOCHS,
        lr     = RESNET_LR,
        tag    = EVAL_TAG,
    )
    print(f"\n  ✅ ResNet addestrata. Checkpoint migliore: {ckpt_p3}")

    # ── STEP 7: Valutazione finale sul test set ───────────────────────────────
    _banner("STEP 7 — Valutazione finale sul test set")
    os.makedirs(METRICS_DIR, exist_ok=True)

    report, cm = evaluate_on_test(
        model_p3, ckpt_p3, test_loader,
        class_names = classes,
        device      = device,
        tag         = EVAL_TAG,
        out_dir     = METRICS_DIR,
    )

    # ── Riepilogo ─────────────────────────────────────────────────────────────
    _banner("RIEPILOGO FINALE")
    acc      = report.get("accuracy", 0.0)
    macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
    print(f"  🏁 Accuracy  : {acc * 100:.2f}%")
    print(f"  🏁 Macro F1  : {macro_f1:.4f}")
    print(f"\n  📊 Report e confusion matrix salvati in: {METRICS_DIR}")
    print(f"  🔖 WandB run : {WANDB_PROJECT} / {WANDB_RUN_NAME}")

    wandb.finish()
    print("\n  ✅ Pipeline completata con successo!\n")


if __name__ == "__main__":
    main()
