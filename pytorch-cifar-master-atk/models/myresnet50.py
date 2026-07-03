import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Cifar10(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Cifar10, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.linear = nn.Linear(self.backbone.fc.out_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.linear(x)
        return x

# Creating an instance of the ResNetCifar10 model
#model = ResNetCifar10(num_classes=10)
