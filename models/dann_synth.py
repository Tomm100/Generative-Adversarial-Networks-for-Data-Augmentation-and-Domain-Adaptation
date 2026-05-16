"""
DANN per Synthetic Domain Adaptation (Synth→Real).

Problema risolto: il classificatore ResNet impara uno "shortcut" tra
le immagini sintetiche NORMAL e la classe NORMAL, ignorando la patologia
reale. Questo causa un crollo della Recall NORMAL al momento del test.

Soluzione: addestrare un DANN che forza il feature extractor a essere
cieco alla differenza reale/sintetica (tramite Gradient Reversal Layer).
In questo modo la rete è costretta a imparare SOLO la patologia.

Setup:
  Source domain (label_domain=0): Immagini REALI (NORMAL + PNEUMONIA)
    → usate per la task loss (classificazione)
    → usate per la domain loss (il discriminatore deve fallire)
  Target domain (label_domain=1): Immagini SINTETICHE NORMAL (dalla GAN)
    → usate SOLO per la domain loss (nessuna label di classe)
    → il GRL forza il feature extractor a non distinguerle dalle reali

Architettura:
  Feature Extractor (G_f):     ResNet-18 → avgpool → 512-dim
  Label Predictor  (G_y):      FC(512 → 2)
  Domain Discriminator (G_d):  MLP(512 → 256 → 128 → 1) + GRL
"""

import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision.models as models


class GradientReversalFunction(Function):
    """
    Forward: identità (x inalterato).
    Backward: moltiplica il gradiente per -lambda_.
    Implementa il minimax tra feature extractor e domain discriminator.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        return -lambda_.item() * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wrapper Module per GradientReversalFunction."""

    def forward(self, x, lambda_=1.0):
        return GradientReversalFunction.apply(x, lambda_)


class DANNSynth(nn.Module):
    """
    Domain-Adversarial Neural Network per Synthetic Domain Adaptation.

    Args:
        num_classes: numero classi task (default: 2 — NORMAL, PNEUMONIA)
        pretrained:  se True, carica pesi ImageNet per il Feature Extractor

    Forward input:
        x:       (B, 3, H, W)
        lambda_: intensità del gradient reversal (0 = nessuno, 1 = pieno)

    Forward output:
        class_logits:  (B, num_classes) — predizione di classe (task)
        domain_logits: (B, 1)           — predizione di dominio (raw logits)
        features:      (B, 512)         — feature estratte (per analisi)
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        # Feature Extractor: ResNet-18 backbone fino ad avgpool
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,     # output: (B, 512, 1, 1)
        )

        # Label Predictor (task principale)
        self.label_predictor = nn.Linear(512, num_classes)

        # Gradient Reversal Layer
        self.grl = GradientReversalLayer()

        # Domain Discriminator
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
        # Feature extraction
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # (B, 512)

        # Task: classificazione NORMAL vs PNEUMONIA
        class_logits = self.label_predictor(features)

        # Domain: Source (reale) vs Target (sintetica)
        reversed_features = self.grl(features, lambda_)
        domain_logits = self.domain_discriminator(reversed_features)

        return class_logits, domain_logits, features
