import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import json
import zipfile


def setup_dataset(drive_path='/content/drive/MyDrive/ProgettoMLVM/modified_dataset',
                  local_dir='./modified_dataset'):
    """
    Carica il modified_dataset da Google Drive.
    Gestisce sia cartella che zip:
      - Se drive_path è una cartella → la usa direttamente
      - Se drive_path + '.zip' esiste → estrae in locale (local_dir)

    Returns:
        (train_dir, val_dir, test_dir) oppure None se non trovato
    """
    zip_path = drive_path + '.zip'

    if os.path.isdir(drive_path):
        # Caso 1: cartella non zippata su Drive
        base = drive_path
        print(f"📁 Dataset trovato come cartella: {drive_path}")

    elif os.path.isfile(zip_path):
        # Caso 2: file .zip su Drive → estrai in locale
        if os.path.exists(local_dir) and os.path.exists(os.path.join(local_dir, 'split_info.json')):
            base = local_dir
            print(f"📁 Dataset già estratto in {local_dir}")
        else:
            print(f"📦 Estrazione {zip_path} → {local_dir}...")
            if os.path.exists(local_dir):
                import shutil
                shutil.rmtree(local_dir)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(local_dir)
            # Controlla se lo zip ha creato una sottocartella
            entries = os.listdir(local_dir)
            if len(entries) == 1 and os.path.isdir(os.path.join(local_dir, entries[0])):
                # Lo zip conteneva una cartella, usa quella
                base = os.path.join(local_dir, entries[0])
            else:
                base = local_dir
            print(f"✅ Estratto in {base}")
    else:
        print(f"ERRORE: né {drive_path} né {zip_path} trovati.")
        print("Esegui prima create_modified_dataset.py e copia su Drive.")
        return None

    train_dir = os.path.join(base, 'train')
    val_dir   = os.path.join(base, 'val')
    test_dir  = os.path.join(base, 'test')

    for d, name in [(train_dir, 'train'), (val_dir, 'val'), (test_dir, 'test')]:
        if not os.path.exists(d):
            print(f"ERRORE: cartella {name}/ non trovata in {base}")
            return None

    # Stampa info
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
    """
    gan_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    gan_dataset = datasets.ImageFolder(root=train_dir, transform=gan_transform)
    gan_loader = DataLoader(gan_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return gan_loader, gan_dataset.classes
