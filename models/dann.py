"""
DANN (Domain-Adversarial Neural Network) per Unsupervised Domain Adaptation.

Architettura:
  - Feature Extractor (G_f): ResNet-18 pretrained → avgpool → 512-dim
  - Label Predictor  (G_y): FC(512 → 2)
  - Domain Discriminator (G_d): MLP(512 → 256 → 128 → 1) + GRL

Il Gradient Reversal Layer (GRL) inverte il gradiente durante la
backpropagation moltiplicandolo per -λ, realizzando il gioco minimax
tra feature extractor e domain discriminator.
"""

import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision.models as models


# ═══════════════════════════════════════════════════════════════
# Gradient Reversal Layer
# ═══════════════════════════════════════════════════════════════

class GradientReversalFunction(Function):
    """
    Funzione autograd custom per il Gradient Reversal.
    Forward: identità (passa x inalterato).
    Backward: moltiplica il gradiente per -λ.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wrapper Module per GradientReversalFunction."""

    def forward(self, x, lambda_=1.0):
        return GradientReversalFunction.apply(x, lambda_)


# ═══════════════════════════════════════════════════════════════
# DANN Model
# ═══════════════════════════════════════════════════════════════

class DANN_Model(nn.Module):
    """
    Domain-Adversarial Neural Network.

    Args:
        num_classes: numero di classi per il Label Predictor (default: 2).

    Forward returns:
        (class_logits, domain_logits, features)
        - class_logits: (B, num_classes) — predizione di classe
        - domain_logits: (B, 1) — predizione di dominio (raw logits, pre-sigmoid)
        - features: (B, 512) — feature estratte dal backbone
    """

    def __init__(self, num_classes=2):
        super().__init__()

        # ── Feature Extractor (G_f): ResNet-18 backbone fino a avgpool ──
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Prendi tutti i layer fino a avgpool (escluso fc)
        self.feature_extractor = nn.Sequential(
            backbone.conv1,      # (B, 64, 112, 112)
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,    # (B, 64, 56, 56)
            backbone.layer1,     # (B, 64, 56, 56)
            backbone.layer2,     # (B, 128, 28, 28)
            backbone.layer3,     # (B, 256, 14, 14)
            backbone.layer4,     # (B, 512, 7, 7)
            backbone.avgpool,    # (B, 512, 1, 1)
        )

        # ── Label Predictor (G_y) ──
        self.label_predictor = nn.Linear(512, num_classes)

        # ── Gradient Reversal Layer ──
        self.grl = GradientReversalLayer()

        # ── Domain Discriminator (G_d) ──
        self.domain_discriminator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x, lambda_=1.0):
        """
        Args:
            x: input tensor (B, 3, 224, 224)
            lambda_: intensità del gradient reversal (0 → nessun reversal, 1 → reversal pieno)

        Returns:
            (class_logits, domain_logits, features)
        """
        # Feature extraction
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # (B, 512)

        # Label prediction (task principale)
        class_logits = self.label_predictor(features)

        # Domain prediction (tramite GRL)
        reversed_features = self.grl(features, lambda_)
        domain_logits = self.domain_discriminator(reversed_features)

        return class_logits, domain_logits, features
