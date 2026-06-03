"""DataLoader per DANN (Domain-Adversarial Neural Network)."""

import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from config import NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS


def _make_balanced_sampler(dataset, num_samples):
    """Crea un WeightedRandomSampler con pesi inversi alla frequenza di classe."""
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
    """Crea i DataLoader per il training DANN."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    source_train = datasets.ImageFolder(
        root=os.path.join(source_dir, 'train'), transform=transform)
    target_train = datasets.ImageFolder(
        root=os.path.join(target_dir, 'train'), transform=transform)
    target_val = datasets.ImageFolder(
        root=os.path.join(target_dir, 'val'), transform=transform)
    target_test = datasets.ImageFolder(
        root=os.path.join(target_dir, 'test'), transform=transform)

    num_samples = max(len(source_train), len(target_train))

    source_sampler = _make_balanced_sampler(source_train, num_samples)
    target_sampler = _make_balanced_sampler(target_train, num_samples)

    source_train_loader = DataLoader(
        source_train, batch_size=batch_size,
        sampler=source_sampler, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS)
    target_train_loader = DataLoader(
        target_train, batch_size=batch_size,
        sampler=target_sampler, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS)
    target_val_loader = DataLoader(
        target_val, batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS)
    target_test_loader = DataLoader(
        target_test, batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS)

    source_labels = [l for _, l in source_train.samples]
    target_labels = [l for _, l in target_train.samples]
    source_counts = np.bincount(source_labels)
    target_counts = np.bincount(target_labels)

    print(f"\n  DANN DataLoader Configuration")
    print(f"  Source (train): {dict(zip(source_train.classes, source_counts))}")
    print(f"  Target (train): {dict(zip(target_train.classes, target_counts))}")
    print(f"  Target (val):   {len(target_val)} samples")
    print(f"  Target (test):  {len(target_test)} samples")
    print(f"  Samples/epoch:  {num_samples}")
    print(f"  Batch/domain:   {batch_size}  (total: {2 * batch_size})")
    print(f"  Image size:     {img_size}x{img_size}")

    return (source_train_loader, target_train_loader,
            target_val_loader, target_test_loader,
            source_train.classes)


def _make_target_test_loader(target_dir, img_size=128, batch_size=32):
    """Crea un DataLoader standalone per il Target test set."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    target_test = datasets.ImageFolder(
        root=os.path.join(target_dir, 'test'), transform=transform)

    target_test_loader = DataLoader(
        target_test, batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS)

    print(f"\n  Target Test Loader ({img_size}x{img_size}): {len(target_test)} samples")

    return target_test_loader, target_test.classes
