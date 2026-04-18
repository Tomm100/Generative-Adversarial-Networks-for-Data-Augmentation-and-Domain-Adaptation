"""
Utilità per la visualizzazione durante il training.
"""

import os
import torch
import matplotlib.pyplot as plt


def save_gan_samples(G, fixed_z, fixed_labels, epoch, samples_dir, num_vis=6):
    """
    Genera e salva una griglia di immagini sintetiche dal Generator.

    Args:
        G:             Generator
        fixed_z:       noise fisso per confrontare le epoche
        fixed_labels:  label condizionali fisse (ignorate per unconditional)
        epoch:         epoca corrente (usata nel titolo e nel nome file)
        samples_dir:   cartella dove salvare le immagini
        num_vis:       numero di immagini nella griglia (unconditional)
    """
    G.eval()
    with torch.no_grad():
        imgs = G(fixed_z, fixed_labels)
    G.train()

    fig, axes = plt.subplots(1, num_vis, figsize=(num_vis * 2.5, 3))
    if num_vis == 1:
        axes = [axes]

    for j in range(num_vis):
        axes[j].imshow(imgs[j, 0].cpu().numpy(), cmap='gray')
        axes[j].axis('off')
        if j == 0:
            axes[j].set_ylabel('NORMAL', fontsize=10)
    
    plt.suptitle(f'GAN Samples (Unconditional) — Epoch {epoch}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(samples_dir, f'samples_epoch_{epoch:03d}.png'))
    plt.close(fig)
