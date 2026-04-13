import torch.nn as nn
import torchvision.models as models


class ResNetClassifier(nn.Module):
    """
    Classificatore basato su ResNet-18 pre-addestrata su ImageNet.
    Sostituisce l'ultimo layer fully-connected per il numero specificato di classi.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        # Carica il backbone ResNet-18 pre-addestrato
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Sostituiamo l'ultimo strato (Fully Connected)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)
