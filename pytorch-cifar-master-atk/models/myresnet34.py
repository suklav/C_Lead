import torch
import torch.nn as nn
import torchvision.models as models

# class ResNet34Cifar10(nn.Module):
#    def __init__(self, num_classes):
#        super(ResNet34Cifar10, self).__init__()
#        self.backbone = models.resnet34(pretrained=False)
#        self.linear = nn.Linear(self.backbone.fc.out_features, num_classes)

#    def forward(self, x):
#        x = self.backbone(x)
#        x = self.linear(x)
#        return x


class ResNet34Cifar10(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet34(pretrained=False)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        in_features = base.fc.in_features
        self.drop = nn.Dropout()
        self.linear = nn.Linear(in_features,num_classes)
    
    def forward(self,x):
        x = self.backbone(x)
        x = self.drop(x.view(-1,self.linear.in_features))
        return self.linear(x)


# Creating an instance of the ResNetCifar10 model
#model = ResNetCifar10(num_classes=10)
