"""
SNGAN: Spectral Normalization GAN con Hinge Loss.

Stessa identica architettura convoluzionale della WGAN-GP (models/wgan.py):
  - Generator: ConvTranspose2d → Upsample+Conv 128x128 con BatchNorm
  - Critic: PatchGAN 128x128 condizionale

Differenze rispetto alla WGAN-GP:
  - Critic: InstanceNorm rimosso → Spectral Norm su tutti i layer Conv
  - Loss:   Wasserstein + GP → Hinge Loss (niente Gradient Penalty)
  - Generator: NESSUNA Spectral Norm (solo BatchNorm, come la WGAN-GP)

Interfaccia 100% compatibile con eval.py:
  G(z, label)   dove z=[B,nz,1,1], label=[B,n_class,1,1] (one-hot)
  D(img, label)  dove img=[B,nc,128,128], label=[B,n_class,128,128] (fill)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def weights_init(m):
    """
    Inizializzazione pesi identica alla WGAN-GP.
    Nota: NON si applica al Critic (i cui pesi sono gestiti dalla SN).
    """
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif 'Norm' in classname and hasattr(m, 'weight') and m.weight is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ═══════════════════════════════════════════════════════════════
# Generator (IDENTICO alla WGAN-GP — nessuna modifica)
# ═══════════════════════════════════════════════════════════════

class SNGenerator(nn.Module):
    """
    Generator 128x128 condizionale.
    Architettura IDENTICA a models/wgan.py Generator.
    Nessuna Spectral Norm: il Generator deve restare libero.

    Flow: z(1x1) + label(1x1) -> 4x4 -> 8 -> 16 -> 32 -> 64 -> 128
    """
    def __init__(self, nz=100, n_class=2, nc=1, d=64):
        super().__init__()
        # 1x1 -> 4x4
        self.deconv1 = nn.ConvTranspose2d(nz + n_class, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)

        # 4x4 -> 8x8
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(d*8, d*4, 3, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(d*4)

        # 8x8 -> 16x16
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(d*4, d*2, 3, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(d*2)

        # 16x16 -> 32x32
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(d*2, d, 3, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d)

        # 32x32 -> 64x64
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = nn.Conv2d(d, d//2, 3, 1, 1)
        self.conv5_bn = nn.BatchNorm2d(d//2)

        # 64x64 -> 128x128
        self.up6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6 = nn.Conv2d(d//2, nc, 3, 1, 1)

    def forward(self, z, label):
        x = torch.cat([z, label], 1)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(self.up2(x))))
        x = F.relu(self.conv3_bn(self.conv3(self.up3(x))))
        x = F.relu(self.conv4_bn(self.conv4(self.up4(x))))
        x = F.relu(self.conv5_bn(self.conv5(self.up5(x))))
        return torch.tanh(self.conv6(self.up6(x)))


# ═══════════════════════════════════════════════════════════════
# Critic (PatchGAN con Spectral Norm — NO InstanceNorm)
# ═══════════════════════════════════════════════════════════════

class SNCritic(nn.Module):
    """
    PatchGAN Critic 128x128 condizionale con Spectral Normalization.

    Differenze rispetto a models/wgan.py Critic:
      - InstanceNorm2d RIMOSSO (incompatibile con Spectral Norm)
      - spectral_norm() applicato a TUTTI i Conv2d
      - Stessa struttura, stesse dimensioni, stesso output 16x16

    Flow (identico alla WGAN-GP):
      128 -> 64(s2) -> 32(s2) -> 16(s2) -> 16(s1) -> 16(s1) -> 16(output)
    """
    def __init__(self, nc=1, n_class=2, d=128):
        super().__init__()
        # 128 -> 64 (stride=2)
        self.conv1 = spectral_norm(nn.Conv2d(nc + n_class, d, 4, 2, 1))

        # 64 -> 32 (stride=2) — NO InstanceNorm
        self.conv2 = spectral_norm(nn.Conv2d(d, d*2, 4, 2, 1))

        # 32 -> 16 (stride=2)
        self.conv3 = spectral_norm(nn.Conv2d(d*2, d*4, 4, 2, 1))

        # 16 -> 16 (stride=1, PatchGAN)
        self.conv4 = spectral_norm(nn.Conv2d(d*4, d*8, 3, 1, 1))

        # 16 -> 16 (stride=1, PatchGAN)
        self.conv5 = spectral_norm(nn.Conv2d(d*8, d*8, 3, 1, 1))

        # 16 -> 16 (output)
        self.conv6 = spectral_norm(nn.Conv2d(d*8, 1, 3, 1, 1))

    def forward(self, img, label):
        x = torch.cat([img, label], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        return self.conv6(x)   # [B, 1, 16, 16]
