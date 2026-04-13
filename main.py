"""
Main script che orchestra l'intera pipeline usando i moduli separati.
"""

import torch
import os
import shutil

from config import (
    DATASET_DIR, RESULTS_DIR, GAN_SAMPLES_DIR, GAN_CHECKPOINTS_DIR,
    SYNTHETIC_DIR, AUGMENTED_DIR,
    RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_EPOCHS, RESNET_LR,
    GAN_IMG_SIZE, GAN_BATCH_SIZE, GAN_EPOCHS, GAN_LR, GAN_N_CRITIC,
    GAN_NZ, GAN_N_CLASS, GAN_NC, GAN_D, GAN_SAVE_EVERY,
    GAN_LR_MILESTONES, GAN_LR_GAMMA,
    SEED,
)
from dataset.loader import setup_dataset, get_dataloaders, get_gan_dataloader
from models.wgan import Generator, Critic, compute_gp
from train import train_resnet, train_wgangp
from eval import evaluate_on_test, generate_synthetic_images, plot_comparison
from utils.seed import set_seed


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Avvio pipeline su Device: {device}")

    # --- 1. SETUP DATASET ---
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        return
    train_dir, val_dir, test_dir = res

    # Conta immagini di training per calcolare il gap da colmare
    n_train_n = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    n_train_p = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    print(f"  Train count: {n_train_n} NORMAL + {n_train_p} PNEUMONIA")

    # --- 2. DATALOADERS ---
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 3. PHASE 1: RESNET BASELINE ---
    model_p1, hist_p1, ckpt_p1 = train_resnet(
        train_loader, val_loader, device,
        epochs=RESNET_EPOCHS, lr=RESNET_LR, tag="Phase1")

    report_p1, cm_p1 = evaluate_on_test(
        model_p1, ckpt_p1, test_loader, classes, device,
        tag="Phase1", out_dir=RESULTS_DIR)

    # --- 4. WGAN-GP TRAINING ---
    gan_loader, gan_classes = get_gan_dataloader(
        train_dir, img_size=GAN_IMG_SIZE, batch_size=GAN_BATCH_SIZE)

    G = Generator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=GAN_D).to(device)
    D = Critic(nc=GAN_NC, n_class=GAN_N_CLASS, d=GAN_D).to(device)

    G, ckpt_gan = train_wgangp(
        G, D, gan_loader, device, compute_gp,
        epochs=GAN_EPOCHS,
        lr=GAN_LR, n_critic=GAN_N_CRITIC, nz=GAN_NZ, n_class=GAN_N_CLASS,
        lr_milestones=GAN_LR_MILESTONES, lr_gamma=GAN_LR_GAMMA,
        save_every=GAN_SAVE_EVERY,
        models_dir=GAN_CHECKPOINTS_DIR,
        samples_dir=GAN_SAMPLES_DIR)

    # --- 5. AUGMENTATION ---
    num_gen_normal = n_train_p - n_train_n     # Colma il gap per NORMAL
    num_gen_pneumonia = 0                      # Nessuna per PNEUMONIA

    print(f"\n{'='*50}\nBILANCIAMENTO DATASET\n{'='*50}")
    print(f"  Gap da colmare: {num_gen_normal} NORMAL sintetiche")
    generate_synthetic_images(
        G, num_gen_normal, num_gen_pneumonia,
        nz=GAN_NZ, n_class=GAN_N_CLASS, device=device, syn_dir=SYNTHETIC_DIR)

    # Crea dataset augmented (train originale + immagini sintetiche)
    aug_train_dir = os.path.join(AUGMENTED_DIR, 'train')
    if os.path.exists(AUGMENTED_DIR):
        shutil.rmtree(AUGMENTED_DIR)
    shutil.copytree(train_dir, aug_train_dir)
    for cat in ['NORMAL', 'PNEUMONIA']:
        sc = os.path.join(SYNTHETIC_DIR, cat)
        if os.path.exists(sc):
            for f in os.listdir(sc):
                shutil.copy(os.path.join(sc, f), os.path.join(aug_train_dir, cat, f))

    print(f"Rapporto Augmented: "
          f"PNEUMONIA:{len(os.listdir(os.path.join(aug_train_dir, 'PNEUMONIA')))} / "
          f"NORMAL:{len(os.listdir(os.path.join(aug_train_dir, 'NORMAL')))}")

    # --- 6. PHASE 3: RESNET AUGMENTED ---
    aug_train_loader, _, _, _ = get_dataloaders(
        aug_train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    model_p3, hist_p3, ckpt_p3 = train_resnet(
        aug_train_loader, val_loader, device,
        epochs=RESNET_EPOCHS, lr=RESNET_LR, tag="Phase3")

    report_p3, cm_p3 = evaluate_on_test(
        model_p3, ckpt_p3, test_loader, classes, device,
        tag="Phase3", out_dir=RESULTS_DIR)

    # --- 7. CONFRONTO FINALE ---
    plot_comparison(hist_p1, hist_p3, cm_p1, cm_p3, classes,
                    report_p1, report_p3, out_dir=RESULTS_DIR)

    print(f"\n📊 Risultati salvati in: {RESULTS_DIR}")
    print("\n Pipeline completata con successo!")


if __name__ == '__main__':
    main()
