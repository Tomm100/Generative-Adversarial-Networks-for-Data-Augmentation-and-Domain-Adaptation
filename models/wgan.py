import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad as torch_grad


def weights_init(m):
    """Inizializzazione pesi per Generator e Critic della WGAN-GP."""
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif 'Norm' in classname and hasattr(m, 'weight') and m.weight is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """Generator condizionale 128x128 per WGAN-GP."""
    def __init__(self, nz=100, n_class=2, nc=1, d=64):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(nz + n_class, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(d*8, d*4, 3, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(d*4)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(d*4, d*2, 3, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(d*2)

        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(d*2, d, 3, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d)

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = nn.Conv2d(d, d//2, 3, 1, 1)
        self.conv5_bn = nn.BatchNorm2d(d//2)

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


class Critic(nn.Module):
    """PatchGAN Critic 128x128 per conditional WGAN-GP. Output: griglia 16x16."""
    def __init__(self, nc=1, n_class=2, d=128):
        super().__init__()
        self.conv1 = nn.Conv2d(nc + n_class, d, 4, 2, 1)

        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_in = nn.InstanceNorm2d(d*2, affine=True)

        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_in = nn.InstanceNorm2d(d*4, affine=True)

        self.conv4 = nn.Conv2d(d*4, d*8, 3, 1, 1)
        self.conv4_in = nn.InstanceNorm2d(d*8, affine=True)

        self.conv5 = nn.Conv2d(d*8, d*8, 3, 1, 1)
        self.conv5_in = nn.InstanceNorm2d(d*8, affine=True)

        self.conv6 = nn.Conv2d(d*8, 1, 3, 1, 1)

    def forward(self, img, label):
        x = torch.cat([img, label], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_in(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_in(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_in(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_in(self.conv5(x)), 0.2)
        return self.conv6(x)


class CriticNoPG(nn.Module):
    """Standard DCGAN Critic 128x128 per conditional WGAN-GP (senza PatchGAN). Output: scalare 1x1."""
    def __init__(self, nc=1, n_class=2, d=128):
        super().__init__()
        self.conv1 = nn.Conv2d(nc + n_class, d, 4, 2, 1)

        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_in = nn.InstanceNorm2d(d*2, affine=True)

        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_in = nn.InstanceNorm2d(d*4, affine=True)

        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_in = nn.InstanceNorm2d(d*8, affine=True)

        self.conv5 = nn.Conv2d(d*8, d*8, 4, 2, 1)
        self.conv5_in = nn.InstanceNorm2d(d*8, affine=True)

        self.conv6 = nn.Conv2d(d*8, 1, 4, 1, 0)

    def forward(self, img, label):
        x = torch.cat([img, label], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_in(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_in(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_in(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_in(self.conv5(x)), 0.2)
        return self.conv6(x)


def compute_gp(D, real, fake, real_lbl, fake_lbl, device, lambda_gp=10):
    """Calcola la Gradient Penalty per WGAN-GP."""
    bs = real.size(0)
    alpha = torch.rand(bs, 1, 1, 1).to(device).expand_as(real)
    interp = (alpha * real.data + (1-alpha) * fake.data).requires_grad_(True)
    interp_l = real_lbl.data

    d_interp = D(interp, interp_l)
    grads = torch_grad(outputs=d_interp, inputs=interp,
                       grad_outputs=torch.ones_like(d_interp),
                       create_graph=True, retain_graph=True)[0]
    grads = grads.view(bs, -1)
    gn = torch.sqrt(torch.sum(grads**2, dim=1) + 1e-12)
    return lambda_gp * ((gn - 1)**2).mean(), gn.mean().item()
