import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import json
import zipfile


def setup_dataset(dataset_dir='./data/modified_dataset'):
    """
    Carica il modified_dataset.
    Gestisce sia cartella che zip:
      - Se dataset_dir è una cartella → la usa direttamente
      - Se dataset_dir + '.zip' esiste → estrae nella stessa posizione

    Returns:
        (train_dir, val_dir, test_dir) oppure None se non trovato
    """
    zip_path = dataset_dir + '.zip'

    if os.path.isdir(dataset_dir):
        base = dataset_dir
        print(f"📁 Dataset trovato: {dataset_dir}")

    elif os.path.isfile(zip_path):
        print(f"📦 Estrazione {zip_path} → {dataset_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dataset_dir)
        # Controlla se lo zip ha creato una sottocartella
        entries = os.listdir(dataset_dir)
        if len(entries) == 1 and os.path.isdir(os.path.join(dataset_dir, entries[0])):
            base = os.path.join(dataset_dir, entries[0])
        else:
            base = dataset_dir
        print(f"✅ Estratto in {base}")
    else:
        print(f"ERRORE: né {dataset_dir} né {zip_path} trovati.")
        print("Assicurati di aver copiato il dataset nella cartella corretta.")
        return None

    train_dir = os.path.join(base, 'train')
    val_dir   = os.path.join(base, 'val')
    test_dir  = os.path.join(base, 'test')

    for d, name in [(train_dir, 'train'), (val_dir, 'val'), (test_dir, 'test')]:
        if not os.path.exists(d):
            print(f"ERRORE: cartella {name}/ non trovata in {base}")
            return None

    # Stampa info split se disponibile
    info_path = os.path.join(base, 'split_info.json')
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        print(f"   Train: {info.get('train_normal', '?')} NORMAL + {info.get('train_pneumonia', '?')} PNEUMONIA")
        print(f"   Val:   {info.get('val_normal', '?')} NORMAL + {info.get('val_pneumonia', '?')} PNEUMONIA")
        print(f"   Test:  {info.get('test_normal', '?')} NORMAL + {info.get('test_pneumonia', '?')} PNEUMONIA")

    return train_dir, val_dir, test_dir


def get_dataloaders(train_dir, val_dir, test_dir, img_size=128, batch_size=16):
    """
    Restituisce i DataLoader per ResNet (RGB convertiti nativamente da ImageFolder).
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes


def get_gan_dataloader(train_dir, img_size=128, batch_size=64):
    """
    Restituisce il DataLoader per il GAN WGAN-GP (1 canale Grayscale, 128x128).
    Utilizza un WeightedRandomSampler per bilanciare i batch a 50/50 tra classi.
    """
    import numpy as np
    from torch.utils.data import WeightedRandomSampler

    gan_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    gan_dataset = datasets.ImageFolder(root=train_dir, transform=gan_transform)

    # Calcola pesi inversi alla frequenza di classe
    labels = [label for _, label in gan_dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts  # peso inversamente proporzionale
    sample_weights = [class_weights[l] for l in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    n_per_class = {gan_dataset.classes[i]: int(c) for i, c in enumerate(class_counts)}
    print(f"  GAN dataloader: {n_per_class} — batch bilanciati 50/50 con WeightedRandomSampler")

    gan_loader = DataLoader(gan_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    return gan_loader, gan_dataset.classes
