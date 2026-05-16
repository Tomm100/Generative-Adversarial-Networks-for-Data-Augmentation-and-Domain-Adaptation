"""
SAGAN: Self-Attention GAN con Spectral Normalization + Hinge Loss.

Differenze chiave rispetto alla WGAN-GP:
  - Spectral Norm su tutti i layer (sostituisce la Gradient Penalty)
  - Self-Attention a 32x32 (coerenza strutturale globale del torace)
  - Conditional Batch Norm nel Generator (modulazione per classe)
  - Projection Discriminator (inner product con class embedding)
  - Hinge Loss (stabilità superiore rispetto a Wasserstein)
  - Residual Blocks (skip connections per gradiente stabile)

Interfaccia Generator compatibile con eval.py:
  G(z, y) dove z=[B,nz,1,1] e y=[B,n_class,1,1] (one-hot) oppure y=[B] (int)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ═══════════════════════════════════════════════════════════════
# Building Blocks
# ═══════════════════════════════════════════════════════════════

class SelfAttention(nn.Module):
    """Self-Attention (Zhang et al., 2019). Cattura dipendenze spaziali globali."""

    def __init__(self, in_ch):
        super().__init__()
        mid = max(in_ch // 8, 1)
        self.query = spectral_norm(nn.Conv2d(in_ch, mid, 1))
        self.key   = spectral_norm(nn.Conv2d(in_ch, mid, 1))
        self.value = spectral_norm(nn.Conv2d(in_ch, in_ch, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # B, HW, C'
        k = self.key(x).view(B, -1, H * W)                      # B, C', HW
        attn = F.softmax(torch.bmm(q, k), dim=-1)               # B, HW, HW
        v = self.value(x).view(B, -1, H * W)                    # B, C, HW
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class ConditionalBatchNorm2d(nn.Module):
    """Conditional BN: modula gain/bias in base alla classe."""

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gain = nn.Embedding(num_classes, num_features)
        self.bias = nn.Embedding(num_classes, num_features)
        nn.init.ones_(self.gain.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x, y):
        g = self.gain(y).unsqueeze(2).unsqueeze(3)
        b = self.bias(y).unsqueeze(2).unsqueeze(3)
        return g * self.bn(x) + b


class GenBlock(nn.Module):
    """Generator residual block: CBN → ReLU → Upsample → Conv."""

    def __init__(self, in_ch, out_ch, n_class):
        super().__init__()
        self.cbn1  = ConditionalBatchNorm2d(in_ch, n_class)
        self.cbn2  = ConditionalBatchNorm2d(out_ch, n_class)
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.up    = nn.Upsample(scale_factor=2, mode='nearest')
        self.skip  = spectral_norm(nn.Conv2d(in_ch, out_ch, 1)) if in_ch != out_ch else nn.Identity()

    def forward(self, x, y):
        h = self.conv1(self.up(F.relu(self.cbn1(x, y))))
        h = self.conv2(F.relu(self.cbn2(h, y)))
        return h + self.skip(self.up(x))


class DiscBlock(nn.Module):
    """Discriminator residual block: ReLU → Conv → Downsample."""

    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.pool  = nn.AvgPool2d(2) if downsample else nn.Identity()
        self.skip  = spectral_norm(nn.Conv2d(in_ch, out_ch, 1)) if in_ch != out_ch else nn.Identity()
        self.skip_pool = nn.AvgPool2d(2) if downsample else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.relu(x))
        h = self.pool(self.conv2(F.relu(h)))
        return h + self.skip_pool(self.skip(x))


class DiscFirstBlock(nn.Module):
    """Primo blocco Discriminator (senza pre-activation)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.pool  = nn.AvgPool2d(2)
        self.skip  = spectral_norm(nn.Conv2d(in_ch, out_ch, 1))

    def forward(self, x):
        h = self.conv1(x)
        h = self.pool(self.conv2(F.relu(h)))
        return h + self.pool(self.skip(x))


# ═══════════════════════════════════════════════════════════════
# Generator
# ═══════════════════════════════════════════════════════════════

class SAGenerator(nn.Module):
    """
    SAGAN Generator (128x128, conditional).

    Architettura: Linear → 4x4 → 8 → 16 → 32(+SA) → 64 → 128
    Con Conditional BN + Residual Blocks + Spectral Norm.

    Args:
        nz:      dimensione vettore latente (default 100)
        n_class: numero classi condizionali (default 2)
        nc:      canali output (default 1, grayscale)
        d:       dimensione base dei canali (default 128)
    """

    def __init__(self, nz=100, n_class=2, nc=1, d=128):
        super().__init__()
        self.nz = nz
        self.n_class = n_class

        # Proiezione iniziale: z → 4x4 feature map
        self.linear = spectral_norm(nn.Linear(nz, (d * 4) * 4 * 4))

        # 4x4 → 8x8 → 16x16 → 32x32(+SA) → 64x64 → 128x128
        self.block1 = GenBlock(d * 4, d * 2, n_class)   # 512→256, 4→8
        self.block2 = GenBlock(d * 2, d,     n_class)   # 256→128, 8→16
        self.block3 = GenBlock(d,     d // 2, n_class)   # 128→64,  16→32
        self.sa     = SelfAttention(d // 2)              # SA a 32x32
        self.block4 = GenBlock(d // 2, d // 4, n_class)  # 64→32,   32→64
        self.block5 = GenBlock(d // 4, d // 4, n_class)  # 32→32,   64→128

        self.final_bn = nn.BatchNorm2d(d // 4)
        self.final_conv = spectral_norm(nn.Conv2d(d // 4, nc, 3, 1, 1))

    def _parse_labels(self, y):
        """Accetta sia one-hot [B, n_class, 1, 1] che integer [B]."""
        if y.dim() == 4:
            y = y.squeeze(-1).squeeze(-1).argmax(dim=1)
        return y.long()

    def forward(self, z, y):
        # Gestisci formati di input diversi
        if z.dim() == 4:
            z = z.squeeze(-1).squeeze(-1)
        y = self._parse_labels(y)

        h = self.linear(z).view(z.size(0), -1, 4, 4)
        h = self.block1(h, y)
        h = self.block2(h, y)
        h = self.block3(h, y)
        h = self.sa(h)
        h = self.block4(h, y)
        h = self.block5(h, y)
        return torch.tanh(self.final_conv(F.relu(self.final_bn(h))))


# ═══════════════════════════════════════════════════════════════
# Discriminator (Projection)
# ═══════════════════════════════════════════════════════════════

class SADiscriminator(nn.Module):
    """
    SAGAN Projection Discriminator (128x128, conditional).

    Architettura: 128 → 64 → 32(+SA) → 16 → 8 → 4 → GlobalPool → scalar
    Con Spectral Norm + Residual Blocks + Projection conditioning.

    Args:
        nc:      canali input (default 1, grayscale)
        n_class: numero classi condizionali (default 2)
        d:       dimensione base dei canali (default 128)
    """

    def __init__(self, nc=1, n_class=2, d=128):
        super().__init__()

        # 128→64→32(+SA)→16→8→4
        self.block1 = DiscFirstBlock(nc, d // 4)         # 1→32,   128→64
        self.block2 = DiscBlock(d // 4, d // 2)          # 32→64,  64→32
        self.sa     = SelfAttention(d // 2)              # SA a 32x32
        self.block3 = DiscBlock(d // 2, d)               # 64→128, 32→16
        self.block4 = DiscBlock(d, d * 2)                # 128→256, 16→8
        self.block5 = DiscBlock(d * 2, d * 4)            # 256→512, 8→4

        # Output: scalar (unconditional) + projection (conditional)
        self.linear = spectral_norm(nn.Linear(d * 4, 1))
        self.embed  = spectral_norm(nn.Embedding(n_class, d * 4))

    def forward(self, x, y):
        """
        Args:
            x: immagine [B, nc, 128, 128]
            y: label classe [B] (integer)
        Returns:
            logits [B, 1]
        """
        h = self.block1(x)
        h = self.block2(h)
        h = self.sa(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = F.relu(h)
        # Global sum pooling
        h = h.sum(dim=[2, 3])  # [B, 512]
        # Projection discrimination: linear(h) + embed(y)^T · h
        out = self.linear(h) + (self.embed(y) * h).sum(dim=1, keepdim=True)
        return out
