"""Main script per SNGAN: Spectral Normalization GAN con Hinge Loss."""


import torch
import os
import shutil
import wandb

from config import (
    DATASET_DIR, RESULTS_DIR, METRICS_DIR,
    RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_EPOCHS, RESNET_LR,
    GAN_NZ, GAN_N_CLASS, GAN_NC,
    GAN_BETA1, GAN_BETA2,
    GAN_DRIVE_BACKUP_EVERY, GAN_DRIVE_DIR,
    SEED,
    # SNGAN Configs
    SNGAN_EPOCHS, SNGAN_LR, SNGAN_N_CRITIC, SNGAN_D,
    SNGAN_IMG_SIZE, SNGAN_BATCH_SIZE,
    SNGAN_SAVE_EVERY, SNGAN_SAMPLES_DIR, SNGAN_CKPT_DIR,
    SNGAN_SYNTH_DIR, SNGAN_AUG_DIR
)
from dataset.loader import setup_dataset, get_dataloaders, get_gan_dataloader
from models.sngan_128 import SNGenerator, SNCritic   # SNGAN 128 Complete: generatore SNGAN + critic PatchGAN
from train import train_resnet, train_sngan
from eval import evaluate_on_test, generate_synthetic_images, plot_comparison
from utils.seed import set_seed




def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Avvio pipeline SNGAN su Device: {device}")

    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        name="SNGAN_Pipeline",
        config={
            "seed":         SEED,
            "architecture": "SNGAN",
            "loss":         "Hinge",
            "normalization":"Spectral Norm (Critic only)",
            "img_size":     SNGAN_IMG_SIZE,
            "batch_size":   SNGAN_BATCH_SIZE,
            "epochs":       SNGAN_EPOCHS,
            "lr":           SNGAN_LR,
            "n_critic":     SNGAN_N_CRITIC,
            "nz":           GAN_NZ,
        }
    )

    wandb.define_metric("Phase1/epoch")
    wandb.define_metric("Phase1/*",  step_metric="Phase1/epoch")
    wandb.define_metric("Phase3/epoch")
    wandb.define_metric("Phase3/*",  step_metric="Phase3/epoch")
    wandb.define_metric("SNGAN/Epoch")
    wandb.define_metric("SNGAN/*",   step_metric="SNGAN/Epoch")


    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        return
    train_dir, val_dir, test_dir = res

    n_train_n = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    n_train_p = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    print(f"  Train: {n_train_n} NORMAL + {n_train_p} PNEUMONIA")
    num_gen_normal    = n_train_p - n_train_n
    num_gen_pneumonia = 0


    train_loader, val_loader, test_loader, classes = get_dataloaders(
        train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)


    print(f"\n{'='*60}\n  PHASE 1: Baseline ResNet (solo dati reali)\n{'='*60}")

    resnet_model, hist_p1, ckpt_p1 = train_resnet(
        train_loader, val_loader, device,
        epochs=RESNET_EPOCHS, lr=RESNET_LR, tag="Phase1")
    report_p1, cm_p1 = evaluate_on_test(
        resnet_model, ckpt_p1, test_loader, classes, device,
        tag="Phase1", out_dir=METRICS_DIR)


    print(f"\n{'='*60}\n  PHASE 2: Training SNGAN\n{'='*60}")

    gan_loader, _ = get_gan_dataloader(
        train_dir, img_size=SNGAN_IMG_SIZE, batch_size=SNGAN_BATCH_SIZE)

    G = SNGenerator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=SNGAN_D).to(device)
    D = SNCritic(nc=GAN_NC, n_class=GAN_N_CLASS, d=SNGAN_D).to(device)

    total_g = sum(p.numel() for p in G.parameters())
    total_d = sum(p.numel() for p in D.parameters())
    print(f"  Generator params:     {total_g:,}")
    print(f"  Critic params:        {total_d:,}")

    G, g_ckpt = train_sngan(
        G, D, gan_loader, device,
        nz=GAN_NZ, n_class=GAN_N_CLASS,
        lr=SNGAN_LR,
        beta1=GAN_BETA1, beta2=GAN_BETA2,
        epochs=SNGAN_EPOCHS, n_critic=SNGAN_N_CRITIC,
        samples_dir=SNGAN_SAMPLES_DIR,
        models_dir=SNGAN_CKPT_DIR,
        save_every=SNGAN_SAVE_EVERY,
        drive_dir=GAN_DRIVE_DIR,
        drive_backup_every=GAN_DRIVE_BACKUP_EVERY,
        use_bagan=True,   # Complete: bilanciamento BAGAN attivo (Standard = False)
    )


    print(f"\n{'='*60}\n  PHASE 3: ResNet su Dataset Augmented (SNGAN)\n{'='*60}")

    generate_synthetic_images(
        G, num_gen_normal, num_gen_pneumonia,
        nz=GAN_NZ, n_class=GAN_N_CLASS, device=device,
        syn_dir=SNGAN_SYNTH_DIR)

    aug_train_dir = os.path.join(SNGAN_AUG_DIR, 'train')
    if os.path.exists(aug_train_dir):
        shutil.rmtree(aug_train_dir)
    shutil.copytree(train_dir, aug_train_dir)

    for cat in ['NORMAL', 'PNEUMONIA']:
        syn_cat = os.path.join(SNGAN_SYNTH_DIR, cat)
        if os.path.exists(syn_cat):
            for f in os.listdir(syn_cat):
                shutil.copy(os.path.join(syn_cat, f),
                            os.path.join(aug_train_dir, cat, f))

    n_aug_n = len(os.listdir(os.path.join(aug_train_dir, 'NORMAL')))
    n_aug_p = len(os.listdir(os.path.join(aug_train_dir, 'PNEUMONIA')))
    print(f"  Augmented: {n_aug_n} NORMAL + {n_aug_p} PNEUMONIA")

    aug_loader, _, _, _ = get_dataloaders(
        aug_train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    resnet_aug, hist_p3, ckpt_p3 = train_resnet(
        aug_loader, val_loader, device,
        epochs=RESNET_EPOCHS, lr=RESNET_LR, tag="Phase3")
    report_p3, cm_p3 = evaluate_on_test(
        resnet_aug, ckpt_p3, test_loader, classes, device,
        tag="Phase3", out_dir=METRICS_DIR)


    plot_comparison(
        hist_p1, hist_p3, cm_p1, cm_p3, classes,
        report_p1, report_p3, out_dir=METRICS_DIR)

    wandb.finish()
    print(f"\n  Pipeline SNGAN completata!")


if __name__ == '__main__':
    main()
