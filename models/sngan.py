"""SNGAN: Spectral Normalization GAN con Hinge Loss — output 256x256."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def weights_init(m):
    """Inizializzazione pesi per il Generator. Non applicare al Critic."""
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif 'Norm' in classname and hasattr(m, 'weight') and m.weight is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class SNGenerator(nn.Module):
    """Generator condizionale 256x256."""
    def __init__(self, nz=100, n_class=2, nc=1, d=128):
        super().__init__()
        self.deconv1    = nn.ConvTranspose2d(nz + n_class, d * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)

        self.up2    = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2  = nn.Conv2d(d * 8, d * 8, 3, 1, 1)
        self.bn2    = nn.BatchNorm2d(d * 8)

        self.up3    = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3  = nn.Conv2d(d * 8, d * 4, 3, 1, 1)
        self.bn3    = nn.BatchNorm2d(d * 4)

        self.up4    = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4  = nn.Conv2d(d * 4, d * 2, 3, 1, 1)
        self.bn4    = nn.BatchNorm2d(d * 2)

        self.up5    = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5  = nn.Conv2d(d * 2, d, 3, 1, 1)
        self.bn5    = nn.BatchNorm2d(d)

        self.up6    = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6  = nn.Conv2d(d, d // 2, 3, 1, 1)
        self.bn6    = nn.BatchNorm2d(d // 2)

        self.up7    = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7  = nn.Conv2d(d // 2, nc, 3, 1, 1)

    def forward(self, z, label):
        x = torch.cat([z, label], 1)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.bn2(self.conv2(self.up2(x))))
        x = F.relu(self.bn3(self.conv3(self.up3(x))))
        x = F.relu(self.bn4(self.conv4(self.up4(x))))
        x = F.relu(self.bn5(self.conv5(self.up5(x))))
        x = F.relu(self.bn6(self.conv6(self.up6(x))))
        return torch.tanh(self.conv7(self.up7(x)))


class SNCritic(nn.Module):
    """PatchGAN Critic 256x256 condizionale con Spectral Normalization. Output: griglia 16x16."""
    def __init__(self, nc=1, n_class=2, d=128):
        super().__init__()
        self.conv0 = spectral_norm(nn.Conv2d(nc + n_class, d // 2, 4, 2, 1))
        self.conv1 = spectral_norm(nn.Conv2d(d // 2, d, 4, 2, 1))
        self.conv2 = spectral_norm(nn.Conv2d(d, d * 2, 4, 2, 1))
        self.conv3 = spectral_norm(nn.Conv2d(d * 2, d * 4, 4, 2, 1))
        self.conv4 = spectral_norm(nn.Conv2d(d * 4, d * 8, 3, 1, 1))
        self.conv5 = spectral_norm(nn.Conv2d(d * 8, d * 8, 3, 1, 1))
        self.conv6 = spectral_norm(nn.Conv2d(d * 8, 1, 3, 1, 1))

    def forward(self, img, label):
        x = torch.cat([img, label], 1)
        x = F.leaky_relu(self.conv0(x), 0.2)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        return self.conv6(x)


class SNCriticNoPG(nn.Module):
    """Standard DCGAN Critic 256x256 condizionale con Spectral Normalization (senza PatchGAN). Output: scalare 1x1."""
    def __init__(self, nc=1, n_class=2, d=128):
        super().__init__()
        self.conv0 = spectral_norm(nn.Conv2d(nc + n_class, d // 2, 4, 2, 1))
        self.conv1 = spectral_norm(nn.Conv2d(d // 2, d, 4, 2, 1))
        self.conv2 = spectral_norm(nn.Conv2d(d, d * 2, 4, 2, 1))
        self.conv3 = spectral_norm(nn.Conv2d(d * 2, d * 4, 4, 2, 1))
        self.conv4 = spectral_norm(nn.Conv2d(d * 4, d * 8, 4, 2, 1))
        self.conv5 = spectral_norm(nn.Conv2d(d * 8, d * 8, 4, 2, 1))
        self.conv6 = spectral_norm(nn.Conv2d(d * 8, 1, 4, 1, 0))

    def forward(self, img, label):
        x = torch.cat([img, label], 1)
        x = F.leaky_relu(self.conv0(x), 0.2)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        return self.conv6(x)
