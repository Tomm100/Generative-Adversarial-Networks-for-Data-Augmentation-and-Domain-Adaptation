"""
DataLoader per DANN (Domain-Adversarial Neural Network).

Crea DataLoader bilanciati per Source e Target domain con:
  - Trasformazioni minimali (no data augmentation): solo Resize, ToTensor, Normalize
  - WeightedRandomSampler per gestire sbilanciamento classi
  - num_samples allineato al dataset più grande → stessa lunghezza iterazione
"""

import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler


def _make_balanced_sampler(dataset, num_samples):
    """
    Crea un WeightedRandomSampler con pesi inversi alla frequenza di classe.

    Args:
        dataset: ImageFolder dataset
        num_samples: numero di campioni da estrarre per epoca

    Returns:
        WeightedRandomSampler
    """
    labels = [label for _, label in dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True
    )


def get_dann_dataloaders(source_dir, target_dir, img_size=224, batch_size=32):
    """
    Crea i DataLoader per il training DANN.

    Vincoli rispettati:
      - Nessuna data augmentation (solo Resize + ToTensor + Normalize)
      - WeightedRandomSampler su entrambi i domini
      - Batch 50/50: ogni step usa un batch Source + un batch Target
      - Normalizzazione con statistiche ImageNet (backbone pretrained)

    Args:
        source_dir: root del Source dataset (con sottocartelle train/, val/, test/)
        target_dir: root del Target dataset (con sottocartelle train/, val/, test/)
        img_size: dimensione immagini (224 per ResNet-18)
        batch_size: batch size per singolo dominio (il batch totale sarà 2×batch_size)

    Returns:
        (source_train_loader, target_train_loader,
         target_val_loader, target_test_loader, class_names)
    """
    # ── Trasformazioni minimali (NO augmentation) ──
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # Statistiche ImageNet
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ── Carica dataset ──
    source_train = datasets.ImageFolder(
        root=os.path.join(source_dir, 'train'), transform=transform)
    target_train = datasets.ImageFolder(
        root=os.path.join(target_dir, 'train'), transform=transform)
    target_val = datasets.ImageFolder(
        root=os.path.join(target_dir, 'val'), transform=transform)
    target_test = datasets.ImageFolder(
        root=os.path.join(target_dir, 'test'), transform=transform)

    # ── Allinea num_samples al dataset più grande ──
    # Questo garantisce che entrambi i DataLoader abbiano la stessa
    # lunghezza di iterazione (nessun dominio si esaurisce prima)
    num_samples = max(len(source_train), len(target_train))

    # ── WeightedRandomSampler per bilanciamento classi ──
    source_sampler = _make_balanced_sampler(source_train, num_samples)
    target_sampler = _make_balanced_sampler(target_train, num_samples)

    # ── DataLoader ──
    source_train_loader = DataLoader(
        source_train, batch_size=batch_size,
        sampler=source_sampler, drop_last=True, num_workers=2
    )
    target_train_loader = DataLoader(
        target_train, batch_size=batch_size,
        sampler=target_sampler, drop_last=True, num_workers=2
    )
    target_val_loader = DataLoader(
        target_val, batch_size=batch_size,
        shuffle=False, num_workers=2
    )
    target_test_loader = DataLoader(
        target_test, batch_size=batch_size,
        shuffle=False, num_workers=2
    )

    # ── Stampa info ──
    source_labels = [l for _, l in source_train.samples]
    target_labels = [l for _, l in target_train.samples]
    source_counts = np.bincount(source_labels)
    target_counts = np.bincount(target_labels)

    print(f"\n  ╔═══════════════════════════════════════════════════╗")
    print(f"  ║           DANN DataLoader Configuration           ║")
    print(f"  ╠═══════════════════════════════════════════════════╣")
    print(f"  ║  Source (train): {dict(zip(source_train.classes, source_counts))}")
    print(f"  ║  Target (train): {dict(zip(target_train.classes, target_counts))}")
    print(f"  ║  Target (val):   {len(target_val)} samples")
    print(f"  ║  Target (test):  {len(target_test)} samples")
    print(f"  ║  Samples/epoch:  {num_samples} (aligned to max)")
    print(f"  ║  Batch/domain:   {batch_size}  (total: {2 * batch_size})")
    print(f"  ║  Image size:     {img_size}×{img_size}")
    print(f"  ║  Normalize:      ImageNet stats")
    print(f"  ╚═══════════════════════════════════════════════════╝")

    return (source_train_loader, target_train_loader,
            target_val_loader, target_test_loader,
            source_train.classes)
