"""Allena una ResNet18 su una cartella di training (reale o augmented) e la valuta.

Logica IDENTICA alla Phase 1 / Phase 3 di experiments/main_sngan.py e main_wgan.py:
    get_dataloaders -> train_resnet -> evaluate_on_test  (stessi parametri).

  --train-dir : cartella di training (default: il train reale del dataset del progetto).
                Passa qui una cartella augmentata (es. prodotta da dataset/build_augmented.py)
                per allenare sul dataset augmentato.
  --tag       : etichetta del run (nome checkpoint best_model_<tag>.pth, chiavi W&B/metriche).

Uso (dalla ROOT del repo):
    python experiments/Resnet_classifier.py                 # baseline su dati reali
    python experiments/Resnet_classifier.py \
        --train-dir ./results/augmented_dataset/train --tag Augmented
"""

import os
import sys
import argparse
import torch
import wandb

# Root del repo nel path: consente l'esecuzione da experiments/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATASET_DIR, METRICS_DIR,
    RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_EPOCHS, RESNET_LR,
    SEED,
)
from dataset.loader import setup_dataset, get_dataloaders
from training.train import train_resnet
from evaluation.eval import evaluate_on_test
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Allena e valuta una ResNet18 su una cartella di training (reale o augmented).")
    parser.add_argument("--train-dir", default=None,
                        help="Cartella di training (default: il train reale del progetto).")
    parser.add_argument("--tag", default="Classifier",
                        help="Etichetta del run (checkpoint best_model_<tag>.pth, chiavi W&B).")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        return
    real_train_dir, val_dir, test_dir = res
    train_dir = args.train_dir or real_train_dir

    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        name=f"resnet_classifier_{args.tag}",
        config={
            "seed":              SEED,
            "train_dir":         train_dir,
            "tag":               args.tag,
            "resnet_img_size":   RESNET_IMG_SIZE,
            "resnet_batch_size": RESNET_BATCH_SIZE,
            "resnet_epochs":     RESNET_EPOCHS,
            "resnet_lr":         RESNET_LR,
        },
    )
    wandb.define_metric(f"{args.tag}/epoch")
    wandb.define_metric(f"{args.tag}/*", step_metric=f"{args.tag}/epoch")

    n_n = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    n_p = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    print(f"  Train dir: {train_dir}")
    print(f"  Train: {n_n} NORMAL + {n_p} PNEUMONIA")

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    print(f"\n{'='*60}\n  Training ResNet ({args.tag})\n{'='*60}")

    model, hist, ckpt = train_resnet(
        train_loader, val_loader, device,
        epochs=RESNET_EPOCHS, lr=RESNET_LR, tag=args.tag)

    report, cm = evaluate_on_test(
        model, ckpt, test_loader, classes, device,
        tag=args.tag, out_dir=METRICS_DIR)

    wandb.finish()
    print(f"\n  Training ResNet ({args.tag}) completato! Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
