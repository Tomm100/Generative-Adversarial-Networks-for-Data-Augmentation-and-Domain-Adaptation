"""
Main script che orchestra l'intera pipeline usando i moduli separati.
"""

import torch
import os
import shutil
import wandb

from config import (
    DATASET_DIR, RESULTS_DIR, METRICS_DIR, GAN_SAMPLES_DIR, GAN_CHECKPOINTS_DIR,
    SYNTHETIC_DIR, AUGMENTED_DIR,
    RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_EPOCHS, RESNET_LR,
    GAN_IMG_SIZE, GAN_BATCH_SIZE, GAN_EPOCHS, GAN_LR, GAN_N_CRITIC,
    GAN_NZ, GAN_N_CLASS, GAN_NC, GAN_D, GAN_SAVE_EVERY,
    GAN_BETA1, GAN_BETA2, GAN_D_WEIGHT_DECAY,
    GAN_EPSILON_PENALTY_COEFF, GAN_WEIGHT_INIT_MEAN, GAN_WEIGHT_INIT_STD,
    GAN_NUM_VIS_SAMPLES, GAN_GEN_BATCH_SIZE, GAN_JPEG_QUALITY,
    GAN_VALIDATE_EVERY, GAN_VAL_RESNET_EPOCHS,
    GAN_DRIVE_BACKUP_EVERY, GAN_DRIVE_DIR,
    NUM_WORKERS, PIN_MEMORY,
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

    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        config={
            "seed":                    SEED,
            # ResNet
            "resnet_img_size":         RESNET_IMG_SIZE,
            "resnet_batch_size":       RESNET_BATCH_SIZE,
            "resnet_epochs":           RESNET_EPOCHS,
            "resnet_lr":               RESNET_LR,
            # WGAN-GP
            "gan_img_size":            GAN_IMG_SIZE,
            "gan_batch_size":          GAN_BATCH_SIZE,
            "gan_epochs":              GAN_EPOCHS,
            "gan_lr":                  GAN_LR,
            "gan_beta1":               GAN_BETA1,
            "gan_beta2":               GAN_BETA2,
            "gan_d_weight_decay":      GAN_D_WEIGHT_DECAY,
            "gan_epsilon_penalty":     GAN_EPSILON_PENALTY_COEFF,
            "gan_weight_init_mean":    GAN_WEIGHT_INIT_MEAN,
            "gan_weight_init_std":     GAN_WEIGHT_INIT_STD,
            "gan_n_critic":            GAN_N_CRITIC,
            "gan_nz":                  GAN_NZ,
            "gan_num_vis_samples":     GAN_NUM_VIS_SAMPLES,
            "gan_gen_batch_size":      GAN_GEN_BATCH_SIZE,
            "gan_jpeg_quality":        GAN_JPEG_QUALITY,
            "gan_validate_every":      GAN_VALIDATE_EVERY,
            "gan_val_resnet_epochs":   GAN_VAL_RESNET_EPOCHS,
            "gan_drive_backup_every":  GAN_DRIVE_BACKUP_EVERY,
            # DataLoader
            "num_workers":             NUM_WORKERS,
            "pin_memory":              PIN_MEMORY,
        }
    )


    wandb.define_metric("Phase1/epoch")
    wandb.define_metric("Phase1/*",         step_metric="Phase1/epoch")
    wandb.define_metric("Phase3/epoch")
    wandb.define_metric("Phase3/*",         step_metric="Phase3/epoch")
    wandb.define_metric("GAN_Training/Epoch")
    wandb.define_metric("GAN_Training/*",   step_metric="GAN_Training/Epoch")
    wandb.define_metric("GAN_Val_TSTR/epoch")
    wandb.define_metric("GAN_Val_TSTR/*",   step_metric="GAN_Val_TSTR/epoch")


    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        return
    train_dir, val_dir, test_dir = res


    n_train_n = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    n_train_p = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    print(f"  Train count: {n_train_n} NORMAL + {n_train_p} PNEUMONIA")

    num_gen_normal = n_train_p - n_train_n
    num_gen_pneumonia = 0


    train_loader, val_loader, test_loader, classes = get_dataloaders(
        train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*60}\n  PHASE 1: Baseline ResNet (solo dati reali)\n{'='*60}")

    model_p1, hist_p1, ckpt_p1 = train_resnet(
        train_loader, val_loader, device,
        epochs=RESNET_EPOCHS, lr=RESNET_LR, tag="Phase1")

    report_p1, cm_p1 = evaluate_on_test(
        model_p1, ckpt_p1, test_loader, classes, device,
        tag="Phase1", out_dir=METRICS_DIR)


    print(f"\n{'='*60}\n  PHASE 2: Training WGAN-GP\n{'='*60}")
    gan_loader, gan_classes = get_gan_dataloader(
        train_dir, img_size=GAN_IMG_SIZE, batch_size=GAN_BATCH_SIZE)

    G = Generator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=GAN_D).to(device)
    D = Critic(nc=GAN_NC, n_class=GAN_N_CLASS, d=GAN_D).to(device)

    total_g = sum(p.numel() for p in G.parameters())
    total_d = sum(p.numel() for p in D.parameters())
    print(f"  Generator params:     {total_g:,}")
    print(f"  Critic params:        {total_d:,}")

    wandb.watch(G, log="all", log_freq=10)
    wandb.watch(D, log="all", log_freq=10)

    G, ckpt_gan = train_wgangp(
        G, D, gan_loader, device, compute_gp,
        epochs=GAN_EPOCHS,
        lr=GAN_LR, n_critic=GAN_N_CRITIC, nz=GAN_NZ, n_class=GAN_N_CLASS,
        beta1=GAN_BETA1, beta2=GAN_BETA2, d_weight_decay=GAN_D_WEIGHT_DECAY,
        save_every=GAN_SAVE_EVERY,
        models_dir=GAN_CHECKPOINTS_DIR,
        samples_dir=GAN_SAMPLES_DIR,

        drive_backup_every=GAN_DRIVE_BACKUP_EVERY,
        drive_dir=GAN_DRIVE_DIR)


    print(f"\n{'='*60}\n  PHASE 3: ResNet su Dataset Augmented (WGAN-GP)\n{'='*60}")

    generate_synthetic_images(
        G, num_gen_normal, num_gen_pneumonia,
        nz=GAN_NZ, n_class=GAN_N_CLASS, device=device, syn_dir=SYNTHETIC_DIR)

    aug_train_dir = os.path.join(AUGMENTED_DIR, 'train')
    if os.path.exists(aug_train_dir):
        shutil.rmtree(aug_train_dir)
    shutil.copytree(train_dir, aug_train_dir)
    
    for cat in ['NORMAL', 'PNEUMONIA']:
        sc = os.path.join(SYNTHETIC_DIR, cat)
        if os.path.exists(sc):
            for f in os.listdir(sc):
                shutil.copy(os.path.join(sc, f), os.path.join(aug_train_dir, cat, f))

    n_aug_n = len(os.listdir(os.path.join(aug_train_dir, 'NORMAL')))
    n_aug_p = len(os.listdir(os.path.join(aug_train_dir, 'PNEUMONIA')))
    print(f"  Augmented: {n_aug_n} NORMAL + {n_aug_p} PNEUMONIA")


    aug_train_loader, _, _, _ = get_dataloaders(
        aug_train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    model_p3, hist_p3, ckpt_p3 = train_resnet(
        aug_train_loader, val_loader, device,
        epochs=RESNET_EPOCHS, lr=RESNET_LR, tag="Phase3")

    report_p3, cm_p3 = evaluate_on_test(
        model_p3, ckpt_p3, test_loader, classes, device,
        tag="Phase3", out_dir=METRICS_DIR)


    plot_comparison(hist_p1, hist_p3, cm_p1, cm_p3, classes,
                    report_p1, report_p3, out_dir=METRICS_DIR)

    wandb.finish()
    print(f"\n  Pipeline WGAN-GP completata!")

if __name__ == '__main__':
    main()
