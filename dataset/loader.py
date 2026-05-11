import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import json
import zipfile
import numpy as np
from config import NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS


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

    Normalizzazione con statistiche ImageNet (mean/std del dataset su cui ResNet
    è pre-addestrata). Questo è fondamentale per evitare uno shock distributivo
    quando i pesi vengono trasferiti al backbone DANN/CDAN, che usa la stessa
    normalizzazione ImageNet.

    NOTA: get_gan_dataloader usa intenzionalmente mean=0.5/std=0.5 perché la
    WGAN-GP opera nel range [-1, 1]. Le due normalizzazioni NON si mescolano:
    il GAN lavora su tensori, mentre il confronto tra immagini reali e generate
    avviene sempre dopo conversione a PNG/JPEG.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # Statistiche ImageNet
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS)

    return train_loader, val_loader, test_loader, train_dataset.classes


def get_gan_dataloader(train_dir, img_size=128, batch_size=64):
    """
    Restituisce il DataLoader per il GAN WGAN-GP (1 canale Grayscale, 128x128).
    Utilizza shuffle=True nel DataLoader.
    """
    
    gan_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    gan_dataset = datasets.ImageFolder(root=train_dir, transform=gan_transform)

    # Calcola il conteggio delle classi per log
    labels = [label for _, label in gan_dataset.samples]
    class_counts = np.bincount(labels)
    n_per_class = {gan_dataset.classes[i]: int(c) for i, c in enumerate(class_counts)}
    print(f"  GAN dataloader: {n_per_class} — shuffle=True (Nessun Oversampling, approccio simil-BAGAN)")

    gan_loader = DataLoader(
        gan_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS)
    return gan_loader, gan_dataset.classes


def get_balanced_train_dataloader(train_dir, img_size=128, batch_size=32):
    """
    Restituisce un DataLoader di training bilanciato tramite WeightedRandomSampler.

    Ogni campione viene estratto con probabilità inversamente proporzionale alla
    frequenza della sua classe, in modo che NORMAL e PNEUMONIA vengano visti
    con uguale frequenza attesa in ogni batch, senza usare dati GAN.

    Il numero di campioni per epoca è pari alla dimensione reale del dataset
    (replacement=True garantisce che la classe minoritaria venga ricampionata).

    Args:
        train_dir:   cartella root del training set (struttura ImageFolder)
        img_size:    dimensione delle immagini (default 128)
        batch_size:  dimensione del batch (default 32)

    Returns:
        (train_loader, class_names)
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # Statistiche ImageNet
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)

    # ── Calcola pesi per campione (inverso della frequenza di classe) ──
    labels = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(labels)
    class_weights_arr = 1.0 / class_counts
    sample_weights = [class_weights_arr[l] for l in labels]

    # num_samples = len dataset → stessa lunghezza di un'epoca normale
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )

    # Stampa info distribuzione
    class_names = train_dataset.classes
    counts_dict = {class_names[i]: int(c) for i, c in enumerate(class_counts)}
    print(f"  [Balanced Loader] Distribuzione reale: {counts_dict}")
    print(f"  [Balanced Loader] WeightedRandomSampler → {len(train_dataset)} campioni/epoca "
          f"(replacement=True)")

    return train_loader, class_names
