"""
DataLoader per DANN Synthetic Domain Adaptation.

Crea due DataLoader allineati:
  - source_loader: immagini REALI (NORMAL + PNEUMONIA) con label di classe
  - target_loader: immagini SINTETICHE NORMAL (dalla GAN) senza label di classe

Entrambi usano WeightedRandomSampler per garantire batch bilanciati.
num_samples allineato al dataset più grande per evitare esaurimento prematuro.

Normalizzazione: ImageNet stats (allineata al backbone ResNet pretrained).
Le immagini sintetiche, salvate come grayscale, vengono convertite in RGB
tramite transforms.Grayscale(num_output_channels=3) per compatibilità.
"""

import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from PIL import Image
from config import NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS, RESNET_IMG_SIZE


def _make_balanced_sampler(dataset, num_samples):
    """
    Crea un WeightedRandomSampler con pesi inversi alla frequenza di classe.
    Garantisce che NORMAL e PNEUMONIA siano viste con uguale probabilità
    in ogni batch (replacement=True per il ricampionamento della classe minore).
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


class SyntheticDataset(Dataset):
    """
    Dataset per le immagini sintetiche generate dalla GAN.

    Carica tutte le immagini da una cartella flat (senza sottocartelle).
    Restituisce (image_tensor, 0) dove 0 è un placeholder per la classe
    (non usato durante il training — il Target non ha supervisione di classe).

    Le immagini sono salvate come JPEG grayscale dalla GAN.
    La trasformazione le converte in RGB (3 canali) per la ResNet.
    """

    def __init__(self, syn_dir, transform=None):
        self.transform = transform
        extensions = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG')
        self.files = [
            os.path.join(syn_dir, f)
            for f in os.listdir(syn_dir)
            if f.endswith(extensions)
        ]
        if not self.files:
            raise RuntimeError(
                f"Nessuna immagine sintetica trovata in: {syn_dir}\n"
                f"Genera prima le immagini con eval.generate_synthetic_images()."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0  # placeholder label (non usato)


def get_synth_dann_loaders(
    real_train_dir,
    synthetic_normal_dir,
    real_val_dir,
    real_test_dir,
    img_size=RESNET_IMG_SIZE,
    batch_size=32
):
    """
    Crea i DataLoader per il training DANN Synth→Real.

    Args:
        real_train_dir:        root del training set reale (ImageFolder: NORMAL/, PNEUMONIA/)
        synthetic_normal_dir:  cartella con le immagini NORMAL sintetiche (flat, no sottocartelle)
        real_val_dir:          cartella val set reale (per model selection durante training)
        real_test_dir:         cartella test set reale (per valutazione finale)
        img_size:              dimensione immagini (default: RESNET_IMG_SIZE dal config)
        batch_size:            batch size per singolo dominio

    Returns:
        (source_loader, target_loader, val_loader, test_loader, class_names)
        - source_loader: reali, bilanciato, con label di classe
        - target_loader: sintetici, senza label di classe utile
        - val_loader:    reali, shuffle=False, per validazione
        - test_loader:   reali, shuffle=False, per valutazione finale
        - class_names:   ['NORMAL', 'PNEUMONIA']
    """
    # Trasformazione condivisa per tutti i loader
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Source: immagini reali con label
    source_ds = datasets.ImageFolder(root=real_train_dir, transform=transform)

    # Target: immagini sintetiche (solo NORMAL, senza label utile)
    target_ds = SyntheticDataset(syn_dir=synthetic_normal_dir, transform=transform)

    # Validation e Test: reali, nessun campionamento
    val_ds  = datasets.ImageFolder(root=real_val_dir,  transform=transform)
    test_ds = datasets.ImageFolder(root=real_test_dir, transform=transform)

    # Allinea num_samples al dataset più grande per iterazione consistente
    num_samples = max(len(source_ds), len(target_ds))

    # Sampler bilanciato per il Source (reale)
    source_sampler = _make_balanced_sampler(source_ds, num_samples)

    # Sampler uniforme per il Target (sintetico: tutte NORMAL, già mono-classe)
    target_sampler = WeightedRandomSampler(
        weights=[1.0] * len(target_ds),
        num_samples=num_samples,
        replacement=True
    )

    source_loader = DataLoader(
        source_ds, batch_size=batch_size,
        sampler=source_sampler, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )
    target_loader = DataLoader(
        target_ds, batch_size=batch_size,
        sampler=target_sampler, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )

    # Log info
    src_labels = [l for _, l in source_ds.samples]
    src_counts = np.bincount(src_labels)
    print(f"\n  ╔═══════════════════════════════════════════════════╗")
    print(f"  ║      DANN Synth→Real DataLoader Configuration     ║")
    print(f"  ╠═══════════════════════════════════════════════════╣")
    print(f"  ║  Source (reali train):  {dict(zip(source_ds.classes, src_counts))}")
    print(f"  ║  Target (sintetici):    {len(target_ds)} immagini NORMAL")
    print(f"  ║  Val (reali):           {len(val_ds)} campioni")
    print(f"  ║  Test (reali):          {len(test_ds)} campioni")
    print(f"  ║  Samples/epoch:         {num_samples} (allineato al max)")
    print(f"  ║  Batch/dominio:         {batch_size}  (totale: {2 * batch_size})")
    print(f"  ║  Image size:            {img_size}×{img_size}")
    print(f"  ╚═══════════════════════════════════════════════════╝")

    return source_loader, target_loader, val_loader, test_loader, source_ds.classes
