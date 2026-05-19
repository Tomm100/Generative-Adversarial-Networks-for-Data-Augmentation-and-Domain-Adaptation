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
        
        # 1. Congela tutti i parametri inizialmente
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 2. Sblocca il layer3 e layer4 (ultimi 2 layer convoluzionali)
        for param in self.backbone.layer3.parameters():
            param.requires_grad = True
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        # 3. Sostituiamo l'ultimo strato (Fully Connected) - il nuovo layer creato ha requires_grad=True di default
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)
