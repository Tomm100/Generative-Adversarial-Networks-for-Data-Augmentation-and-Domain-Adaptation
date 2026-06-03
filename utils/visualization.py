"""Utilita per la visualizzazione durante il training."""

import os
import torch
import matplotlib.pyplot as plt


def save_gan_samples(G, fixed_z, fixed_labels, epoch, samples_dir, num_vis=6):
    """Genera e salva una griglia di immagini sintetiche dal Generator."""
    G.eval()
    with torch.no_grad():
        imgs = G(fixed_z, fixed_labels)
        imgs = (imgs + 1) / 2
    G.train()

    fig, axes = plt.subplots(2, num_vis, figsize=(num_vis * 2.5, 5))
    for cls_idx, cls_name in enumerate(['NORMAL', 'PNEUMONIA']):
        for j in range(num_vis):
            idx = cls_idx * num_vis + j
            axes[cls_idx, j].imshow(imgs[idx, 0].cpu().numpy(), cmap='gray')
            axes[cls_idx, j].axis('off')
            if j == 0:
                axes[cls_idx, j].set_ylabel(cls_name, fontsize=10)
    plt.suptitle(f'GAN Samples — Epoch {epoch}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(samples_dir, f'samples_epoch_{epoch:03d}.png'))
    plt.close(fig)
