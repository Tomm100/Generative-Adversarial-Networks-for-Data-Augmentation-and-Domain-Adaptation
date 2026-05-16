"""
Training loop per SAGAN con Hinge Loss + Spectral Normalization.

Differenze chiave rispetto a train_wgangp (train.py):
  - Hinge Loss invece di Wasserstein + GP (no gradient penalty)
  - Spectral Norm gestisce il vincolo di Lipschitz (nei layer del modello)
  - TTUR: D ha LR 4x superiore a G (Two Time-scale Update Rule)
  - n_critic = 1 (standard per Hinge Loss + SN)
  - Projection Discriminator (etichette come inner product, non concatenazione)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm

from config import (
    GAN_NUM_VIS_SAMPLES, GAN_SAVE_EVERY,
    GAN_DRIVE_BACKUP_EVERY, GAN_DRIVE_DIR,
    GAN_VALIDATE_EVERY, GAN_VAL_RESNET_EPOCHS,
    RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_LR,
    AUGMENTED_DIR,
)


def save_sagan_samples(G, fixed_z, fixed_labels, epoch, out_dir, num_vis):
    """Salva una griglia di campioni generati (identica alla versione WGAN)."""
    G.eval()
    with torch.no_grad():
        imgs = G(fixed_z, fixed_labels)
        imgs = (imgs + 1) / 2  # [-1,1] → [0,1]

    fig, axes = plt.subplots(2, num_vis, figsize=(num_vis * 2, 4))
    fig.suptitle(f'GAN Samples — Epoch {epoch}', fontsize=14)
    for i in range(2):
        for j in range(num_vis):
            idx = i * num_vis + j
            axes[i, j].imshow(imgs[idx].squeeze().cpu().numpy(), cmap='gray')
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'samples_epoch_{epoch:03d}.png'), dpi=100)
    plt.close(fig)
    G.train()


def train_sagan(
    G, D, gan_loader, device,
    nz=100, n_class=2,
    lr_g=1e-4, lr_d=4e-4,
    beta1=0.0, beta2=0.999,
    epochs=300, n_critic=1,
    samples_dir='./results/sagan_samples',
    models_dir='./results/sagan_checkpoints',
    save_every=10,
    drive_dir=None,
    drive_backup_every=50,
    # Validazione TSTR
    validate_every=0,
    resnet_epochs=5,
    train_dir=None, val_dir=None,
    num_gen_normal=0, num_gen_pneumonia=0,
    resnet_img_size=128, resnet_batch_size=32,
    augmented_dir='./results/augmented_dataset_sagan',
):
    """
    Training loop SAGAN con Hinge Loss.

    Hinge Loss:
      D_loss = E[max(0, 1 - D(real))] + E[max(0, 1 + D(fake))]
      G_loss = -E[D(fake)]

    Returns:
        (G, g_final_path, best_val_epoch, val_history)
    """
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Validazione TSTR opzionale
    val_enabled = (validate_every > 0
                   and train_dir is not None
                   and val_dir is not None)

    # TTUR: Discriminator ha LR più alto del Generator
    G_opt = optim.Adam(G.parameters(), lr=lr_g, betas=(beta1, beta2))
    D_opt = optim.Adam(D.parameters(), lr=lr_d, betas=(beta1, beta2))

    # LR decay lineare (come nella WGAN-GP)
    G_sched = optim.lr_scheduler.LinearLR(G_opt, start_factor=1.0, end_factor=0.0, total_iters=epochs)
    D_sched = optim.lr_scheduler.LinearLR(D_opt, start_factor=1.0, end_factor=0.0, total_iters=epochs)

    # Fixed noise per visualizzazione
    num_vis = GAN_NUM_VIS_SAMPLES
    fixed_z = torch.randn(num_vis * 2, nz, device=device)  # [12, 100]
    fixed_labels = torch.cat([
        torch.zeros(num_vis, dtype=torch.long, device=device),
        torch.ones(num_vis, dtype=torch.long, device=device)
    ])  # [12]

    # Tracking
    val_history = []
    best_val_f1 = -1.0
    best_val_epoch = 0

    print(f"\nTraining SAGAN: {epochs} epoche (Hinge Loss + Spectral Norm)")
    print(f"  LR_G: {lr_g}, LR_D: {lr_d} (TTUR), n_critic={n_critic}")
    if val_enabled:
        print(f"  Validazione ogni {validate_every} epoche")

    gan_start = time.time()

    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []
        d_real_scores, d_fake_scores = [], []

        pbar = tqdm(gan_loader, desc=f"  Epoch {epoch}/{epochs}", leave=False)
        for batch_idx, (x_real, y_real) in enumerate(pbar):
            mb = x_real.size(0)
            x_real = x_real.to(device)
            y_real = y_real.to(device)

            # ═══════════════════════════════════════════
            # Train Discriminator (Hinge Loss)
            # ═══════════════════════════════════════════
            D.zero_grad()

            # Real
            d_real = D(x_real, y_real)  # [B, 1]
            loss_real = torch.relu(1.0 - d_real).mean()

            # Fake
            z = torch.randn(mb, nz, device=device)
            # Genera con le stesse label del batch reale (come nella WGAN-GP)
            fake = G(z, y_real)
            d_fake = D(fake.detach(), y_real)
            loss_fake = torch.relu(1.0 + d_fake).mean()

            d_loss = loss_real + loss_fake
            d_loss.backward()
            D_opt.step()

            d_losses.append(d_loss.item())
            d_real_scores.append(d_real.mean().item())
            d_fake_scores.append(d_fake.mean().item())

            # ═══════════════════════════════════════════
            # Train Generator (ogni n_critic step)
            # ═══════════════════════════════════════════
            if (batch_idx + 1) % n_critic == 0:
                G.zero_grad()
                z = torch.randn(mb, nz, device=device)

                # 50/50 bilanciamento label (come nella WGAN-GP)
                half = mb // 2
                y_gen = torch.cat([
                    torch.zeros(half, dtype=torch.long),
                    torch.ones(mb - half, dtype=torch.long)
                ])[torch.randperm(mb)].to(device)

                g_loss = -D(G(z, y_gen), y_gen).mean()
                g_loss.backward()
                G_opt.step()
                g_losses.append(g_loss.item())

            pbar.set_postfix({
                'D': f"{d_loss.item():.3f}",
                'G': f"{g_losses[-1] if g_losses else 0:.3f}"
            })

        # LR Scheduler
        G_sched.step()
        D_sched.step()

        # Metriche epoca
        elapsed = (time.time() - gan_start) / 60
        eta = (elapsed / epoch) * (epochs - epoch)
        d_avg = np.mean(d_losses)
        g_avg = np.mean(g_losses) if g_losses else 0
        d_real_avg = np.mean(d_real_scores)
        d_fake_avg = np.mean(d_fake_scores)

        log_dict = {
            "SAGAN/Discriminator_Loss": d_avg,
            "SAGAN/Generator_Loss":     g_avg,
            "SAGAN/D_real_score":       d_real_avg,
            "SAGAN/D_fake_score":       d_fake_avg,
            "SAGAN/LR_Generator":       G_opt.param_groups[0]['lr'],
            "SAGAN/Epoch":              epoch,
        }

        # Stampa e campioni visivi
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{epoch}/{epochs}] D: {d_avg:.3f} | G: {g_avg:.3f} | "
                  f"D(real): {d_real_avg:.2f} | D(fake): {d_fake_avg:.2f} | "
                  f"Time: {elapsed:.1f}m | ETA: {eta:.1f}m")
            save_sagan_samples(G, fixed_z, fixed_labels, epoch, samples_dir, num_vis)
            sample_path = os.path.join(samples_dir, f'samples_epoch_{epoch:03d}.png')
            if os.path.exists(sample_path):
                log_dict["SAGAN/Generated_Images"] = wandb.Image(sample_path, caption=f"Epoch {epoch}")

        wandb.log(log_dict)

        # Checkpoint locale
        if epoch % save_every == 0 or epoch == epochs:
            g_path = os.path.join(models_dir, f'G_epoch_{epoch}.pth')
            d_path = os.path.join(models_dir, f'D_epoch_{epoch}.pth')
            torch.save(G.state_dict(), g_path)
            torch.save(D.state_dict(), d_path)
            print(f"  [Checkpoint] G + D salvati (ep.{epoch})")

            # Drive backup
            if (drive_dir and drive_backup_every > 0
                    and epoch % drive_backup_every == 0):
                if os.path.isdir(os.path.dirname(drive_dir)) or os.path.exists(drive_dir):
                    import shutil
                    os.makedirs(drive_dir, exist_ok=True)
                    shutil.copy(g_path, os.path.join(drive_dir, f'G_sagan_epoch_{epoch}.pth'))
                    shutil.copy(d_path, os.path.join(drive_dir, f'D_sagan_epoch_{epoch}.pth'))
                    print(f"  [Drive Backup] ep.{epoch} → {drive_dir}")

        # Validazione TSTR periodica
        if val_enabled and epoch % validate_every == 0:
            val_result = _run_sagan_validation(
                G, epoch, device, nz, n_class,
                train_dir, val_dir,
                num_gen_normal, num_gen_pneumonia,
                resnet_img_size, resnet_batch_size, resnet_epochs,
                augmented_dir, best_val_f1)
            val_history.append(val_result)
            if val_result['macro_f1'] > best_val_f1:
                best_val_f1 = val_result['macro_f1']
                best_val_epoch = epoch

    # Fine training
    gan_time = (time.time() - gan_start) / 60
    g_final = os.path.join(models_dir, f'G_epoch_{epochs}.pth')
    print(f"\nSAGAN training completato in {gan_time:.1f} minuti!")

    if val_history:
        print(f"\n{'='*50}\nValidation Summary\n{'='*50}")
        for r in val_history:
            marker = " ← BEST" if r['epoch'] == best_val_epoch else ""
            print(f"  Ep.{r['epoch']:3d} | F1: {r['macro_f1']:.4f} | Acc: {r['accuracy']:.2f}%{marker}")

    return G, g_final, best_val_epoch, val_history


def _run_sagan_validation(G, epoch, device, nz, n_class,
                          train_dir, val_dir,
                          num_gen_normal, num_gen_pneumonia,
                          resnet_img_size, resnet_batch_size, resnet_epochs,
                          augmented_dir, current_best_f1):
    """Validazione TSTR identica a quella della WGAN-GP (riusa eval.py)."""
    import shutil
    from eval import generate_synthetic_images
    from dataset.loader import get_dataloaders
    from train import train_resnet

    print(f"\n  {'─'*40}\n  [VAL] Validazione SAGAN epoca {epoch}\n  {'─'*40}")

    val_tmp = './_val_tmp_sagan'
    tmp_syn = os.path.join(val_tmp, 'synthetic')
    tmp_aug = os.path.join(val_tmp, 'augmented', 'train')

    if os.path.exists(val_tmp):
        shutil.rmtree(val_tmp)

    generate_synthetic_images(
        G, num_gen_normal, num_gen_pneumonia,
        nz=nz, n_class=n_class, device=device, syn_dir=tmp_syn)

    shutil.copytree(train_dir, tmp_aug)
    for cat in ['NORMAL', 'PNEUMONIA']:
        syn_cat = os.path.join(tmp_syn, cat)
        if os.path.exists(syn_cat):
            for f in os.listdir(syn_cat):
                shutil.copy(os.path.join(syn_cat, f),
                            os.path.join(tmp_aug, cat, f))

    aug_loader, val_loader, _, _ = get_dataloaders(
        tmp_aug, val_dir, val_dir,
        img_size=resnet_img_size, batch_size=resnet_batch_size)

    _, hist, _ = train_resnet(
        aug_loader, val_loader, device,
        epochs=resnet_epochs, lr=RESNET_LR, tag=f"SAGAN_Val_ep{epoch}")

    best_f1 = max(hist['val_macro_f1'])
    best_acc = hist['val_acc'][hist['val_macro_f1'].index(best_f1)]
    result = {'epoch': epoch, 'macro_f1': best_f1, 'accuracy': best_acc}

    wandb.log({
        "SAGAN_Val_TSTR/macro_f1": best_f1,
        "SAGAN_Val_TSTR/accuracy": best_acc,
        "SAGAN_Val_TSTR/epoch": epoch
    })
    print(f"  [VAL] Ep.{epoch}: Macro F1 = {best_f1:.4f}, Acc = {best_acc:.2f}%")

    if best_f1 > current_best_f1:
        print(f"  [VAL] ★ Nuovo miglior F1! Salvo dataset augmented...")
        if os.path.exists(augmented_dir):
            shutil.rmtree(augmented_dir)
        shutil.copytree(tmp_aug, os.path.join(augmented_dir, 'train'))

    if os.path.exists(val_tmp):
        shutil.rmtree(val_tmp)

    return result
