"""
SNGAN: Spectral Normalization GAN con Hinge Loss — output 256x256.

Generator:  z(1x1) → 4 → 8 → 16 → 32 → 64 → 128 → 256
Critic:     256 → 128(s2) → 64(s2) → 32(s2) → 16(s2) → 16(s1) → 16(s1) → 16(output)

Differenze rispetto alla versione 128px:
  - Generator: aggiunto blocco up7 (128→256)
  - Critic:    aggiunto conv0 (256→128, stride=2) come primo layer
  - Interfaccia invariata: G(z, label), D(img, label)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def weights_init(m):
    """
    Inizializzazione pesi per il Generator.
    NON applicare al Critic: i pesi sono gestiti dalla Spectral Norm.
    """
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif 'Norm' in classname and hasattr(m, 'weight') and m.weight is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class SNGenerator(nn.Module):
    """
    Generator condizionale 256x256.

    Flow: z(1x1) + label(1x1) → 4 → 8 → 16 → 32 → 64 → 128 → 256

    Args:
        nz:      dimensione dello spazio latente
        n_class: numero di classi (per la label one-hot condizionale)
        nc:      canali output (1 = grayscale)
        d:       larghezza base dei canali (128 consigliato)
    """
    def __init__(self, nz=100, n_class=2, nc=1, d=128):
        super().__init__()
        # 1x1 → 4x4
        self.deconv1    = nn.ConvTranspose2d(nz + n_class, d * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)

        # 4 → 8
        self.up2    = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2  = nn.Conv2d(d * 8, d * 8, 3, 1, 1)
        self.bn2    = nn.BatchNorm2d(d * 8)

        # 8 → 16
        self.up3    = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3  = nn.Conv2d(d * 8, d * 4, 3, 1, 1)
        self.bn3    = nn.BatchNorm2d(d * 4)

        # 16 → 32
        self.up4    = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4  = nn.Conv2d(d * 4, d * 2, 3, 1, 1)
        self.bn4    = nn.BatchNorm2d(d * 2)

        # 32 → 64
        self.up5    = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5  = nn.Conv2d(d * 2, d, 3, 1, 1)
        self.bn5    = nn.BatchNorm2d(d)

        # 64 → 128
        self.up6    = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6  = nn.Conv2d(d, d // 2, 3, 1, 1)
        self.bn6    = nn.BatchNorm2d(d // 2)

        # 128 → 256
        self.up7    = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7  = nn.Conv2d(d // 2, nc, 3, 1, 1)

    def forward(self, z, label):
        x = torch.cat([z, label], 1)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))   # 4x4
        x = F.relu(self.bn2(self.conv2(self.up2(x))))   # 8x8
        x = F.relu(self.bn3(self.conv3(self.up3(x))))   # 16x16
        x = F.relu(self.bn4(self.conv4(self.up4(x))))   # 32x32
        x = F.relu(self.bn5(self.conv5(self.up5(x))))   # 64x64
        x = F.relu(self.bn6(self.conv6(self.up6(x))))   # 128x128
        return torch.tanh(self.conv7(self.up7(x)))       # 256x256


class SNCritic(nn.Module):
    """
    PatchGAN Critic 256x256 condizionale con Spectral Normalization.

    Flow: 256 → 128(s2) → 64(s2) → 32(s2) → 16(s2) → 16(s1) → 16(s1) → 16(output)
    Output: griglia PatchGAN 16x16 (invariata rispetto alla versione 128px).

    Args:
        nc:      canali input (1 = grayscale)
        n_class: numero di classi (per la fill-map condizionale)
        d:       larghezza base dei canali
    """
    def __init__(self, nc=1, n_class=2, d=128):
        super().__init__()
        # 256 → 128 (stride=2) — layer aggiunto per 256px
        self.conv0 = spectral_norm(nn.Conv2d(nc + n_class, d // 2, 4, 2, 1))

        # 128 → 64 (stride=2)
        self.conv1 = spectral_norm(nn.Conv2d(d // 2, d, 4, 2, 1))

        # 64 → 32 (stride=2)
        self.conv2 = spectral_norm(nn.Conv2d(d, d * 2, 4, 2, 1))

        # 32 → 16 (stride=2)
        self.conv3 = spectral_norm(nn.Conv2d(d * 2, d * 4, 4, 2, 1))

        # --- ABLATION: SENZA PATCHGAN (Standard DCGAN) ---
        # 16x16 -> 8x8
        # self.conv4 = spectral_norm(nn.Conv2d(d * 4, d * 8, 4, 2, 1))
        # 8x8 -> 4x4
        # self.conv5 = spectral_norm(nn.Conv2d(d * 8, d * 8, 4, 2, 1))
        # 4x4 -> 1x1
        # self.conv6 = spectral_norm(nn.Conv2d(d * 8, 1, 4, 1, 0))

        # (Codice PatchGAN originale ripristinato)
        self.conv4 = spectral_norm(nn.Conv2d(d * 4, d * 8, 3, 1, 1))
        self.conv5 = spectral_norm(nn.Conv2d(d * 8, d * 8, 3, 1, 1))
        self.conv6 = spectral_norm(nn.Conv2d(d * 8, 1, 3, 1, 1))

    def forward(self, img, label):
        x = torch.cat([img, label], 1)
        x = F.leaky_relu(self.conv0(x), 0.2)   # 128x128
        x = F.leaky_relu(self.conv1(x), 0.2)   # 64x64
        x = F.leaky_relu(self.conv2(x), 0.2)   # 32x32
        x = F.leaky_relu(self.conv3(x), 0.2)   # 16x16
        
        # --- ABLATION: SENZA PATCHGAN ---
        # x = F.leaky_relu(self.conv4(x), 0.2)   # 8x8
        # x = F.leaky_relu(self.conv5(x), 0.2)   # 4x4
        # return self.conv6(x)                   # [B, 1, 1, 1]

        # (Codice PatchGAN originale ripristinato)
        x = F.leaky_relu(self.conv4(x), 0.2)   # 16x16
        x = F.leaky_relu(self.conv5(x), 0.2)   # 16x16
        return self.conv6(x)                   # [B, 1, 16, 16]
