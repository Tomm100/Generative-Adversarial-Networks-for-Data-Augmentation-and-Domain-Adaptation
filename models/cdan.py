"""
CDAN (Conditional Adversarial Domain Adaptation) per Unsupervised Domain Adaptation.

Architettura:
  - Feature Extractor (G_f): ResNet-18 pretrained → avgpool → 512-dim
  - Label Predictor  (G_y): FC(512 → 2) → softmax probabilities
  - Domain Discriminator (G_d): MLP(1024 → 512 → 256 → 1) + GRL
    Input: prodotto multilineare (outer product) tra features (512) e
           softmax probs (2), appiattito a 1024 = 512 × 2.

Il condizionamento multilineare (Long et al., 2018) permette al discriminatore
di dominio di ricevere informazioni sia sullo spazio delle feature che sulle
predizioni del classificatore, allineando così le distribuzioni condizionate
p(f | y) invece delle sole distribuzioni marginali p(f).

Il GRL viene applicato all'input del discriminatore (già nello spazio 1024-dim),
realizzando il gioco minimax tra feature extractor e discriminatore.

Riferimento:
  Long et al., "Conditional Adversarial Domain Adaptation", NeurIPS 2018.
  https://arxiv.org/abs/1705.10667
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.models as models


# ═══════════════════════════════════════════════════════════════
# Gradient Reversal Layer (identico a models/dann.py, copiato
# per mantenere cdan.py completamente autocontenuto)
# ═══════════════════════════════════════════════════════════════

class GradientReversalFunction(Function):
    """
    Funzione autograd custom per il Gradient Reversal.
    Forward:  identità (x inalterato).
    Backward: moltiplica il gradiente in ingresso per -λ.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # Nessun gradiente rispetto a lambda_ (scalare non-tensor)
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wrapper Module per GradientReversalFunction."""

    def forward(self, x, lambda_=1.0):
        return GradientReversalFunction.apply(x, lambda_)


# ═══════════════════════════════════════════════════════════════
# CDAN Model
# ═══════════════════════════════════════════════════════════════

class CDAN_Model(nn.Module):
    """
    Conditional Adversarial Domain Adaptation Model.

    Differenza chiave rispetto a DANN:
      - Il discriminatore di dominio riceve il PRODOTTO MULTILINEARE tra
        le feature f (512-dim) e le probabilità softmax ŷ (2-dim), ottenendo
        un tensore condizionato di dimensione 1024 = 512 × 2.
      - Questo forza l'allineamento delle distribuzioni condizionate p(f | y),
        non solo delle distribuzioni marginali p(f) come in DANN.

    Il GRL viene inserito subito prima del discriminatore, applicato al
    vettore multilineare (1024-dim) già costruito.

    Args:
        num_classes: numero di classi per il Label Predictor (default: 2).
        feature_dim: dimensione delle feature del backbone (default: 512).

    Forward returns (training mode):
        (class_logits, domain_logits, features, softmax_probs)
        - class_logits:   (B, num_classes) — logit di classificazione
        - domain_logits:  (B, 1)           — logit di dominio (pre-sigmoid)
        - features:       (B, 512)         — feature raw del backbone
        - softmax_probs:  (B, num_classes) — probabilità softmax del classificatore
    """

    def __init__(self, num_classes: int = 2, feature_dim: int = 512):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        # Dimensione del vettore multilineare: feature_dim × num_classes
        self.multilinear_dim = feature_dim * num_classes  # 512 × 2 = 1024

        # ── Feature Extractor (G_f): ResNet-18 backbone fino ad avgpool ──
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(
            backbone.conv1,      # (B, 64, H/2, W/2)
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,    # (B, 64, H/4, W/4)
            backbone.layer1,     # (B, 64,  ...)
            backbone.layer2,     # (B, 128, ...)
            backbone.layer3,     # (B, 256, ...)
            backbone.layer4,     # (B, 512, ...)
            backbone.avgpool,    # (B, 512, 1, 1)
        )

        # ── Label Predictor (G_y) ──
        self.label_predictor = nn.Linear(feature_dim, num_classes)

        # ── Gradient Reversal Layer ──
        self.grl = GradientReversalLayer()

        # ── Domain Discriminator (G_d) ──
        # Input: 1024-dim (prodotto multilineare features × softmax)
        # Architettura leggermente più profonda di DANN per gestire l'input più grande.
        self.domain_discriminator = nn.Sequential(
            nn.Linear(self.multilinear_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    # ─── Metodo utility: prodotto multilineare ─────────────────────────────
    def _multilinear_map(self, features: torch.Tensor,
                         softmax_probs: torch.Tensor) -> torch.Tensor:
        """
        Calcola il prodotto multilineare (outer product) tra features e probs,
        appiattito a un vettore 1D per campione.

        Implementazione efficiente con torch.bmm:
          - features:      (B, D)  →  (B, D, 1)
          - softmax_probs: (B, C)  →  (B, 1, C)
          - bmm result:    (B, D, C)
          - flatten:       (B, D*C)

        Args:
            features:     (B, feature_dim)   = (B, 512)
            softmax_probs:(B, num_classes)    = (B, 2)

        Returns:
            multilinear:  (B, feature_dim * num_classes) = (B, 1024)
        """
        B = features.size(0)
        # Espandiamo per bmm: (B, 512, 1) × (B, 1, 2) → (B, 512, 2)
        f = features.unsqueeze(2)          # (B, 512, 1)
        p = softmax_probs.unsqueeze(1)     # (B, 1, 2)
        outer = torch.bmm(f, p)            # (B, 512, 2)
        return outer.view(B, -1)           # (B, 1024)

    # ─── Forward ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, lambda_: float = 1.0):
        """
        Args:
            x:       input tensor (B, 3, H, W) — immagini normalizzate
            lambda_: intensità del gradient reversal ∈ [0, 1]
                     (0 → nessun reversal durante warm-up, 1 → reversal pieno)

        Returns:
            (class_logits, domain_logits, features, softmax_probs)
        """
        # ── 1. Feature Extraction ──
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)   # (B, 512)

        # ── 2. Label Prediction ──
        class_logits = self.label_predictor(features)    # (B, 2)
        softmax_probs = F.softmax(class_logits, dim=1)   # (B, 2)  — detached per evitare
        # Nota: NON si esegue detach() sui softmax_probs: vogliamo che i gradienti
        # fluiscano nel feature extractor sia via task loss che via multilinear map.
        # Il GRL gestisce la direzione del gradiente per il discriminatore.

        # ── 3. Prodotto Multilineare (Conditional Coupling) ──
        multilinear = self._multilinear_map(features, softmax_probs)  # (B, 1024)

        # ── 4. Gradient Reversal + Domain Discrimination ──
        reversed_multilinear = self.grl(multilinear, lambda_)
        domain_logits = self.domain_discriminator(reversed_multilinear)  # (B, 1)

        return class_logits, domain_logits, features, softmax_probs
