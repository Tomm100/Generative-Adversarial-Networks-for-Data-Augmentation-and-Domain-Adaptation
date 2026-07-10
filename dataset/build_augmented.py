"""Costruisce il dataset AUGMENTED a partire dai pesi di una GAN e da una
percentuale di gap da colmare.

SOLO data-prep: genera i sintetici NORMAL, li unisce al train reale e restituisce
il path della train dir augmentata. Nessun training, nessuna valutazione.

Riusa `generate_synthetic_images` (evaluation/eval.py); la copia/unione e' quella
della Phase 3 di main_sngan.py / main_wgan.py.

Uso (dalla ROOT del repo):
    python dataset/build_augmented.py --gan-ckpt G_epoch_210.pth --gap 75
    python dataset/build_augmented.py --gan-ckpt G.pth --gap 100 --gan sngan128 \
        --out-dir ./results/augmented_dataset --synth-dir ./results/augmented_synth

Da notebook:
    from dataset.build_augmented import build_augmented_dataset
    aug_train_dir = build_augmented_dataset("G.pth", gap_percent=75, ...)
"""

import os
import sys
import shutil
import argparse
import torch

# Root del repo nel path: consente l'esecuzione da dataset/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATASET_DIR, RESULTS_DIR, GAN_NZ, GAN_N_CLASS, GAN_NC, GAN_D, SEED,
)
from dataset.loader import setup_dataset
from evaluation.eval import generate_synthetic_images
from utils.seed import set_seed


def _load_generator(gan_type, ckpt_path, device):
    """Istanzia il generatore corretto e carica i pesi dal checkpoint."""
    if gan_type == "sngan128":
        from models.sngan_128 import SNGenerator as Generator
    elif gan_type == "wgan":
        from models.wgan import Generator
    else:
        raise ValueError(f"--gan non valido: '{gan_type}'. Usa 'sngan128' o 'wgan'.")
    G = Generator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=GAN_D).to(device)
    G.load_state_dict(torch.load(ckpt_path, map_location=device))
    return G


def build_augmented_dataset(gan_ckpt, gap_percent, out_dir, synth_dir,
                            gan_type="sngan128", reuse=True, device=None):
    """Crea la train dir augmentata (reale + gap% di sintetici NORMAL).

    Ritorna il path della cartella <out_dir>/train.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        raise RuntimeError("Dataset non trovato (controlla DATASET_DIR in config.py).")
    train_dir, _, _ = res

    n_normal = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    n_pneum  = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    max_deficit = n_pneum - n_normal
    num_synth = int(max_deficit * (gap_percent / 100.0))
    print(f"Train reale: {n_normal} NORMAL + {n_pneum} PNEUMONIA | gap = {max_deficit}")
    print(f"Gap da colmare: {gap_percent}% -> {num_synth} sintetici NORMAL")

    # ── Generazione sintetici NORMAL (riusa se gia' presenti) ──
    syn_normal_dir = os.path.join(synth_dir, 'NORMAL')
    have = len(os.listdir(syn_normal_dir)) if os.path.isdir(syn_normal_dir) else 0
    if reuse and have >= num_synth:
        print(f"  [synth] gia' presenti {have} immagini in {syn_normal_dir}. Skip generazione.")
    else:
        print(f"  [synth] genero {num_synth} NORMAL da {gan_ckpt} ({gan_type})...")
        G = _load_generator(gan_type, gan_ckpt, device)
        generate_synthetic_images(
            G, num_gen_normal=num_synth, num_gen_pneumonia=0,
            nz=GAN_NZ, n_class=GAN_N_CLASS, device=device, syn_dir=synth_dir)
        del G
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Unione: copia il train reale + inserisci i sintetici NORMAL ──
    aug_train_dir = os.path.join(out_dir, 'train')
    if os.path.exists(aug_train_dir):
        shutil.rmtree(aug_train_dir)
    shutil.copytree(train_dir, aug_train_dir)

    dest_normal = os.path.join(aug_train_dir, 'NORMAL')
    synth_files = sorted(os.listdir(syn_normal_dir))[:num_synth]
    for f in synth_files:
        shutil.copy(os.path.join(syn_normal_dir, f), os.path.join(dest_normal, f))

    n_aug_n = len(os.listdir(dest_normal))
    n_aug_p = len(os.listdir(os.path.join(aug_train_dir, 'PNEUMONIA')))
    print(f"Dataset augmentato: {n_aug_n} NORMAL + {n_aug_p} PNEUMONIA -> {aug_train_dir}")
    return aug_train_dir


def main():
    parser = argparse.ArgumentParser(
        description="Costruisce il dataset augmented (reale + gap% di sintetici NORMAL).")
    parser.add_argument("--gan-ckpt", required=True, help="Checkpoint del generatore GAN (.pth).")
    parser.add_argument("--gap", type=float, default=100.0,
                        help="Percentuale del gap Normal/Pneumonia da colmare (default: 100).")
    parser.add_argument("--gan", choices=["sngan128", "wgan"], default="sngan128",
                        help="Architettura del generatore per caricare il ckpt.")
    parser.add_argument("--out-dir", default=os.path.join(RESULTS_DIR, "augmented_dataset"),
                        help="Cartella di output (verra' creata <out-dir>/train).")
    parser.add_argument("--synth-dir", default=os.path.join(RESULTS_DIR, "augmented_synth"),
                        help="Cartella dei sintetici (generati o riusati).")
    parser.add_argument("--no-reuse", action="store_true",
                        help="Rigenera i sintetici anche se gia' presenti.")
    args = parser.parse_args()

    set_seed(SEED)
    build_augmented_dataset(
        gan_ckpt=args.gan_ckpt, gap_percent=args.gap,
        out_dir=args.out_dir, synth_dir=args.synth_dir,
        gan_type=args.gan, reuse=not args.no_reuse)


if __name__ == "__main__":
    main()
