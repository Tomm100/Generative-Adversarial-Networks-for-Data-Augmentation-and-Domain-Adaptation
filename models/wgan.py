import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad as torch_grad


def weights_init(m):
    """
    Inizializzazione rigorosa dei pesi per Generator e Critic della WGAN-GP.
    Da usare con: G.apply(weights_init) e D.apply(weights_init).

    Logica:
      - Conv2d / ConvTranspose2d : pesi ~ N(0.0, 0.02),  bias = 0
      - InstanceNorm2d / BatchNorm2d (affine=True):
            weight (γ) ~ N(1.0, 0.02)   ← media 1 per non annullare la normalizzazione
            bias   (β) = 0
    """
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif 'Norm' in classname and hasattr(m, 'weight') and m.weight is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    Generator 128x128 per conditional WGAN-GP.
    Flow: z(1x1) + label(1x1) -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128

    Modifiche rispetto alla versione originale:
    - Il primo layer (1x1 -> 4x4) rimane ConvTranspose2d perché è una proiezione
      lineare dallo spazio latente allo spazio spaziale (non produce checkerboard).
    - Tutti i layer successivi usano Upsample(nearest) + Conv2d(3x3) per evitare
      i checkerboard artifacts tipici della ConvTranspose2d.

    Conditional input:
      z:     [B, nz, 1, 1]       rumore latente
      label: [B, n_class, 1, 1]  one-hot spaziale (creato con torch.eye)
      → concatenati lungo dim=1 → [B, nz + n_class, 1, 1]
    """
    def __init__(self, nz=100, n_class=2, nc=1, d=64):
        super().__init__()
        # 1x1 -> 4x4 (proiezione latente, ConvTranspose2d è corretto qui)
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
        # Concatena rumore e label spazialmente
        # z: [B, nz, 1, 1], label: [B, n_class, 1, 1] -> [B, nz+n_class, 1, 1]
        x = torch.cat([z, label], 1)

        x = F.relu(self.deconv1_bn(self.deconv1(x)))   # [B, d*8, 4, 4]
        x = F.relu(self.conv2_bn(self.conv2(self.up2(x))))   # [B, d*4, 8, 8]
        x = F.relu(self.conv3_bn(self.conv3(self.up3(x))))   # [B, d*2, 16, 16]
        x = F.relu(self.conv4_bn(self.conv4(self.up4(x))))   # [B, d,   32, 32]
        x = F.relu(self.conv5_bn(self.conv5(self.up5(x))))   # [B, d//2,64, 64]
        return torch.tanh(self.conv6(self.up6(x)))            # [B, nc,  128,128]


class Critic(nn.Module):
    """
    PatchGAN Critic 128x128 per conditional WGAN-GP.

    Output: griglia 16x16 — ogni cella valuta un patch locale dell'immagine,
    forzando il Generator a curare le texture ad alta frequenza ovunque.

    Flow: 128→64(s2) → 32(s2) → 16(s2) → 16(s1) → 16(s1) → 16(output)

    Conditional input:
      img:   [B, nc, 128, 128]       immagine (grayscale)
      label: [B, n_class, 128, 128]  spatial label map (fill tensor)
      → concatenati lungo dim=1 → [B, nc + n_class, 128, 128]
    """
    def __init__(self, nc=1, n_class=2, d=128):
        super().__init__()
        # 128 -> 64 (stride=2)
        self.conv1 = nn.Conv2d(nc + n_class, d, 4, 2, 1)

        # 64 -> 32 (stride=2)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_in = nn.InstanceNorm2d(d*2, affine=True)

        # 32 -> 16 (stride=2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_in = nn.InstanceNorm2d(d*4, affine=True)

        # 16 -> 16 (stride=1, PatchGAN)
        self.conv4 = nn.Conv2d(d*4, d*8, 3, 1, 1)
        self.conv4_in = nn.InstanceNorm2d(d*8, affine=True)

        # 16 -> 16 (stride=1, PatchGAN)
        self.conv5 = nn.Conv2d(d*8, d*8, 3, 1, 1)
        self.conv5_in = nn.InstanceNorm2d(d*8, affine=True)

        # 16 -> 16 (output)
        self.conv6 = nn.Conv2d(d*8, 1, 3, 1, 1)

    def forward(self, img, label):
        x = torch.cat([img, label], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)                 # [B, d,   64, 64]
        x = F.leaky_relu(self.conv2_in(self.conv2(x)), 0.2)  # [B, d*2, 32, 32]
        x = F.leaky_relu(self.conv3_in(self.conv3(x)), 0.2)  # [B, d*4, 16, 16]
        x = F.leaky_relu(self.conv4_in(self.conv4(x)), 0.2)  # [B, d*8, 16, 16]
        x = F.leaky_relu(self.conv5_in(self.conv5(x)), 0.2)  # [B, d*8, 16, 16]
        return self.conv6(x)                                  # [B, 1,   16, 16]


class CriticNoPG(nn.Module):
    """
    Standard DCGAN Critic 128x128 per conditional WGAN-GP (senza PatchGAN).

    Output: scalare 1x1 — giudizio globale sull'intera immagine.

    Flow: 128→64(s2) → 32(s2) → 16(s2) → 8(s2) → 4(s2) → 1(output)

    Conditional input:
      img:   [B, nc, 128, 128]       immagine (grayscale)
      label: [B, n_class, 128, 128]  spatial label map (fill tensor)
      → concatenati lungo dim=1 → [B, nc + n_class, 128, 128]
    """
    def __init__(self, nc=1, n_class=2, d=128):
        super().__init__()
        # 128 -> 64 (stride=2)
        self.conv1 = nn.Conv2d(nc + n_class, d, 4, 2, 1)

        # 64 -> 32 (stride=2)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_in = nn.InstanceNorm2d(d*2, affine=True)

        # 32 -> 16 (stride=2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_in = nn.InstanceNorm2d(d*4, affine=True)

        # 16 -> 8 (stride=2)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_in = nn.InstanceNorm2d(d*8, affine=True)

        # 8 -> 4 (stride=2)
        self.conv5 = nn.Conv2d(d*8, d*8, 4, 2, 1)
        self.conv5_in = nn.InstanceNorm2d(d*8, affine=True)

        # 4 -> 1 (output)
        self.conv6 = nn.Conv2d(d*8, 1, 4, 1, 0)

    def forward(self, img, label):
        x = torch.cat([img, label], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)                 # [B, d,   64, 64]
        x = F.leaky_relu(self.conv2_in(self.conv2(x)), 0.2)  # [B, d*2, 32, 32]
        x = F.leaky_relu(self.conv3_in(self.conv3(x)), 0.2)  # [B, d*4, 16, 16]
        x = F.leaky_relu(self.conv4_in(self.conv4(x)), 0.2)  # [B, d*8,  8,  8]
        x = F.leaky_relu(self.conv5_in(self.conv5(x)), 0.2)  # [B, d*8,  4,  4]
        return self.conv6(x)                                  # [B, 1,    1,  1]


def compute_gp(D, real, fake, real_lbl, fake_lbl, device, lambda_gp=10):
    """
    Calcola la Gradient Penalty per WGAN-GP.

    Compatibile sia con Critic scalare (1x1) che PatchGAN (NxN):
    - torch.ones_like(d_interp) si adatta automaticamente alla shape dell'output.
    - I gradienti vengono appiattiti e la norma L2 viene calcolata per-sample.
    """
    bs = real.size(0)
    alpha = torch.rand(bs, 1, 1, 1).to(device).expand_as(real)
    interp = (alpha * real.data + (1-alpha) * fake.data).requires_grad_(True)

    # CRITICAL FIX C-WGAN: We do NOT interpolate labels!
    # Interpolating discrete map conditionals breaks the Critic assumption
    # and ruins the W_dist gradient leading to huge instability.
    interp_l = real_lbl.data

    d_interp = D(interp, interp_l)
    grads = torch_grad(outputs=d_interp, inputs=interp,
                       grad_outputs=torch.ones_like(d_interp),
                       create_graph=True, retain_graph=True)[0]
    grads = grads.view(bs, -1)
    gn = torch.sqrt(torch.sum(grads**2, dim=1) + 1e-12)
    return lambda_gp * ((gn - 1)**2).mean(), gn.mean().item()
