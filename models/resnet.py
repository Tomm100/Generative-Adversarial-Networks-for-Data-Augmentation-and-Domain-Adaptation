import torch.nn as nn
import torchvision.models as models


class ResNetClassifier(nn.Module):
    """Classificatore basato su ResNet-18 pre-addestrata su ImageNet."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        for param in self.backbone.layer3.parameters():
            param.requires_grad = True
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)
