"""
Script minimale: carica i pesi di una ResNet già addestrata e calcola
la Macro F1 (e il classification report completo) sul test set.
Riutilizza le funzioni già presenti nel progetto.
"""

import torch
from sklearn.metrics import classification_report
import torch.nn.functional as F

from config import (
    DATASET_DIR, RESNET_IMG_SIZE, RESNET_BATCH_SIZE, SEED
)
from dataset.loader import setup_dataset, get_dataloaders
from models.resnet import ResNetClassifier
from utils.seed import set_seed

# ==============================================================================
# CONFIGURAZIONE — MODIFICA QUI
# ==============================================================================
RESNET_CKPT = "/content/drive/MyDrive/ProgettoMLVM/checkpoints/best_model_Ablation_75pct.pth"
# ==============================================================================


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Verifica checkpoint ──
    import os
    if not os.path.exists(RESNET_CKPT):
        print(f"ERRORE: Checkpoint non trovato: {RESNET_CKPT}")
        return

    # ── Dataset ──
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        print("ERRORE: dataset non trovato.")
        return
    train_dir, val_dir, test_dir = res

    _, _, test_loader, class_names = get_dataloaders(
        train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE
    )

    # ── Caricamento modello ──
    print(f"\nCaricamento pesi da: {RESNET_CKPT}")
    model = ResNetClassifier(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(RESNET_CKPT, map_location=device))
    model.eval()
    print("Pesi caricati correttamente.")

    # ── Inferenza ──
    print("\nInferenza sul test set...")
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            _, pred = torch.max(logits, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())

    # ── Risultati ──
    report_str  = classification_report(all_labels, all_preds, target_names=class_names)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    print(f"\n{'='*50}")
    print("  RISULTATI SUL TEST SET")
    print(f"{'='*50}")
    print(report_str)
    print(f"  Macro F1:  {report_dict['macro avg']['f1-score']:.4f}")
    print(f"  Accuracy:  {report_dict['accuracy']:.4f}")


if __name__ == "__main__":
    main()
