"""
CDAN per Synthetic Domain Adaptation (Synth→Real).

Evoluzione della DANN: il domain discriminator è condizionato sul prodotto
tensoriale multilineare tra feature e predizioni di classe (Long et al., 2018).
Questo allinea le distribuzioni condizionali P(f|y) anziché le marginali P(f),
preservando la struttura discriminativa tra classi.

Differenze chiave rispetto alla DANN:
  - Il discriminatore riceve f ⊗ softmax(g) anziché f
  - Entropy Conditioning (CDAN+E): campioni con alta entropia pesano meno
    nella domain loss, stabilizzando il training su predizioni incerte

Setup (identico alla DANN):
  Source domain (label_domain=0): Immagini REALI (NORMAL + PNEUMONIA)
  Target domain (label_domain=1): Immagini SINTETICHE NORMAL (dalla GAN)

Architettura:
  Feature Extractor (G_f):     ResNet-18 → avgpool → 512-dim
  Label Predictor  (G_y):      FC(512 → 2)
  Multilinear Map:             f ⊗ softmax(g) → 512×2 = 1024-dim
  Domain Discriminator (G_d):  MLP(1024 → 512 → 256 → 1) + GRL

Riferimenti:
  Long et al., "Conditional Adversarial Domain Adaptation", NeurIPS 2018
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


class CDANSynth(nn.Module):
    """
    Conditional Domain-Adversarial Neural Network per Synthetic DA.

    Args:
        num_classes: numero classi task (default: 2 — NORMAL, PNEUMONIA)
        pretrained:  se True, carica pesi ImageNet per il Feature Extractor
        use_entropy: se True, abilita Entropy Conditioning (CDAN+E)

    Forward input:
        x:       (B, 3, H, W)
        lambda_: intensità del gradient reversal (0 = nessuno, 1 = pieno)

    Forward output:
        class_logits:  (B, num_classes) — predizione di classe (task)
        domain_logits: (B, 1)           — predizione di dominio (raw logits)
        features:      (B, 512)         — feature estratte (per analisi UMAP)
    """

    def __init__(self, num_classes=2, pretrained=True, use_entropy=True):
        super().__init__()
        self.num_classes = num_classes
        self.use_entropy = use_entropy
        feat_dim = 512

        # Feature Extractor: ResNet-18 backbone fino ad avgpool
        # Allineato a ResNetClassifier: conv1, bn1, layer1, layer2 CONGELATI
        #                               layer3, layer4 SBLOCCATI
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # 1. Congela tutti i parametri del backbone
        for param in backbone.parameters():
            param.requires_grad = False

        # 2. Sblocca layer3 e layer4 (come in ResNetClassifier)
        for param in backbone.layer3.parameters():
            param.requires_grad = True
        for param in backbone.layer4.parameters():
            param.requires_grad = True

        self.feature_extractor = nn.Sequential(
            backbone.conv1,       # FROZEN
            backbone.bn1,         # FROZEN
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,      # FROZEN
            backbone.layer2,      # FROZEN
            backbone.layer3,      # TRAINABLE
            backbone.layer4,      # TRAINABLE
            backbone.avgpool,     # output: (B, 512, 1, 1)
        )

        # Label Predictor (task principale)
        self.label_predictor = nn.Linear(feat_dim, num_classes)

        # Gradient Reversal Layer
        self.grl = GradientReversalLayer()

        # Domain Discriminator — input condizionato: f ⊗ softmax(g)
        # Con 2 classi: 512 × 2 = 1024 dimensioni
        cond_dim = feat_dim * num_classes  # 1024
        self.domain_discriminator = nn.Sequential(
            nn.Linear(cond_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x, lambda_=1.0):
        # Feature extraction
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # (B, 512)

        # Task: classificazione NORMAL vs PNEUMONIA
        class_logits = self.label_predictor(features)        # (B, num_classes)

        # Multilinear conditioning: f ⊗ softmax(g)
        # Produce una rappresentazione congiunta feature-predizione
        softmax_probs = torch.softmax(class_logits, dim=1)     # (B, num_classes)
        # Outer product: (B, 512, 1) × (B, 1, num_classes) → (B, 512, num_classes)
        # Poi flatten → (B, 512 * num_classes) = (B, 1024)
        cond_input = torch.bmm(
            features.unsqueeze(2),          # (B, 512, 1)
            softmax_probs.unsqueeze(1)        # (B, 1, num_classes)
        ).view(features.size(0), -1)        # (B, 1024)

        # Domain: Source (reale) vs Target (sintetica)
        reversed_cond = self.grl(cond_input, lambda_)
        domain_logits = self.domain_discriminator(reversed_cond)  # (B, 1)

        return class_logits, domain_logits, features, softmax_probs

    @staticmethod
    def compute_entropy_weight(softmax_probs):
        """
        Entropy Conditioning (CDAN+E).
        Campioni con alta entropia (predizione incerta) contribuiscono meno
        alla domain loss → stabilizza il training su campioni ambigui.

        H(p) = -Σ p_i * log(p_i)   ∈ [0, log(num_classes)]
        weight = 1 + exp(-H)        ∈ [1+exp(-log(C)), 2]

        Args:
            softmax_probs: (B, num_classes) — probabilità di classe dal task predictor
        Returns:
            weight: (B, 1) — pesi per la domain loss, normalizzati per batch
        """
        # Entropia per campione: H(p) = -Σ p_i * log(p_i + ε)
        entropy = -torch.sum(
            softmax_probs * torch.log(softmax_probs + 1e-8), dim=1
        )  # (B,)
        # Peso: alta entropia → peso basso (meno contributo)
        weight = 1.0 + torch.exp(-entropy)  # (B,)
        # Normalizzazione per batch (media=1)
        weight = weight / weight.mean()
        return weight.unsqueeze(1)  # (B, 1)
