"""
Script standalone: carica i pesi di una DANNSynth già addestrata e calcola
la Macro F1 (e il classification report completo) sul test set.
Riutilizza le funzioni già presenti nel progetto.

Utilizzo:
    python eval_f1_dann.py --ckpt /path/al/best_DANN_Synth.pth
"""

import argparse
import os
import torch
import wandb

from config import DATASET_DIR, RESNET_IMG_SIZE, RESNET_BATCH_SIZE, METRICS_DIR
from dataset.loader import setup_dataset, get_dataloaders
from models.dann_synth import DANNSynth
from train_dann_synth import evaluate_dann_synth


def main(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # ── Verifica checkpoint ──
    if not os.path.exists(ckpt_path):
        print(f"ERRORE: file dei pesi non trovato: {ckpt_path}")
        return

    # ── Dataset ──
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        print("ERRORE: dataset non trovato.")
        return
    train_dir, val_dir, test_dir = res

    _, _, test_loader, class_names = get_dataloaders(
        train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=32
    )

    # ── W&B (richiesto da evaluate_dann_synth per il log) ──
    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        name="eval_DANN_Synth"
    )

    # ── Valutazione ──
    print(f"\nCaricamento pesi DANN da: {ckpt_path}")
    model = DANNSynth(num_classes=len(class_names), pretrained=False)

    report_dict, cm = evaluate_dann_synth(
        model=model,
        ckpt_path=ckpt_path,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        tag="DANN_Synth",
        out_dir=METRICS_DIR
    )

    print(f"\n  Macro F1 : {report_dict['macro avg']['f1-score']:.4f}")
    print(f"  Accuracy : {report_dict['accuracy']:.4f}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Valuta una DANNSynth già addestrata sul test set."
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Percorso al file dei pesi della DANN (es. best_DANN_Synth.pth)"
    )
    args = parser.parse_args()
    main(args.ckpt)
