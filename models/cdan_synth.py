"""CDAN per Synthetic Domain Adaptation (Synth->Real)."""

import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision.models as models


class GradientReversalFunction(Function):
    """Forward: identita. Backward: moltiplica il gradiente per -lambda_."""

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
    """Conditional Domain-Adversarial Neural Network per Synthetic DA."""

    def __init__(self, num_classes=2, pretrained=True, use_entropy=True):
        super().__init__()
        self.num_classes = num_classes
        self.use_entropy = use_entropy
        feat_dim = 512

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        for param in backbone.parameters():
            param.requires_grad = False

        for param in backbone.layer3.parameters():
            param.requires_grad = True
        for param in backbone.layer4.parameters():
            param.requires_grad = True

        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )

        self.label_predictor = nn.Linear(feat_dim, num_classes)
        self.grl = GradientReversalLayer()

        cond_dim = feat_dim * num_classes
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
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)

        class_logits = self.label_predictor(features)

        softmax_probs = torch.softmax(class_logits, dim=1)
        cond_input = torch.bmm(
            features.unsqueeze(2),
            softmax_probs.unsqueeze(1)
        ).view(features.size(0), -1)

        reversed_cond = self.grl(cond_input, lambda_)
        domain_logits = self.domain_discriminator(reversed_cond)

        return class_logits, domain_logits, features, softmax_probs

    @staticmethod
    def compute_entropy_weight(softmax_probs):
        """Entropy Conditioning (CDAN+E)."""
        entropy = -torch.sum(
            softmax_probs * torch.log(softmax_probs + 1e-8), dim=1
        )
        weight = 1.0 + torch.exp(-entropy)
        weight = weight / weight.mean()
        return weight.unsqueeze(1)
