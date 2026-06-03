"""DANN per Synthetic Domain Adaptation (Synth->Real)."""

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


class DANNSynth(nn.Module):
    """Domain-Adversarial Neural Network per Synthetic Domain Adaptation."""

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

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

        self.label_predictor = nn.Linear(512, num_classes)
        self.grl = GradientReversalLayer()

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
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)

        class_logits = self.label_predictor(features)

        reversed_features = self.grl(features, lambda_)
        domain_logits = self.domain_discriminator(reversed_features)

        return class_logits, domain_logits, features
