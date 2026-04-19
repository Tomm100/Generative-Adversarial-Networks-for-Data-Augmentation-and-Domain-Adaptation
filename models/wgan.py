import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad as torch_grad

class Generator(nn.Module):
    """
    Generator 128x128 per conditional WGAN-GP.
    Flow: z(1x1) -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
    """
    def __init__(self, nz=100, n_class=2, nc=1, d=128):
        super().__init__()
        self.deconv1_1 = nn.ConvTranspose2d(nz, d*4, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*4)
        self.deconv1_2 = nn.ConvTranspose2d(n_class, d*4, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*4)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, d//2, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d//2)
        self.deconv6 = nn.ConvTranspose2d(d//2, nc, 4, 2, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(self._modules[m], (nn.ConvTranspose2d, nn.Conv2d)):
                self._modules[m].weight.data.normal_(mean, std)

    def forward(self, z, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(z)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.relu(self.deconv5_bn(self.deconv5(x)))
        return torch.tanh(self.deconv6(x))


class Critic(nn.Module):
    """
    Critic 128x128 per conditional WGAN-GP.
    Flow: 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4 -> 1x1
    """
    def __init__(self, nc=1, n_class=2, d=128):
        super().__init__()
        self.conv1_1 = nn.Conv2d(nc, d//2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(n_class, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_ln = nn.LayerNorm([d*2, 32, 32])
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_ln = nn.LayerNorm([d*4, 16, 16])
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_ln = nn.LayerNorm([d*8, 8, 8])
        self.conv5 = nn.Conv2d(d*8, d*8, 4, 2, 1)
        self.conv5_ln = nn.LayerNorm([d*8, 4, 4])
        self.conv6 = nn.Conv2d(d*8, 1, 4, 1, 0)

    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(self._modules[m], (nn.ConvTranspose2d, nn.Conv2d)):
                self._modules[m].weight.data.normal_(mean, std)

    def forward(self, img, label):
        x = F.leaky_relu(self.conv1_1(img), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_ln(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_ln(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_ln(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_ln(self.conv5(x)), 0.2)
        return self.conv6(x)

def compute_gp(D, real, fake, real_lbl, fake_lbl, device, lambda_gp=10):
    """
    Calcola la Gradient Penalty per WGAN-GP.
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
