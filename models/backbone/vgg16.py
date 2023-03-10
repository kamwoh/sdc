import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

from models.backbone.base import BaseNet


class VGG16(BaseNet):
    def __init__(self, pretrained=True, **kwargs):
        super(VGG16, self).__init__()

        if pretrained:
            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            model = vgg16()

        self.features = model.features
        self.avgpool = model.avgpool
        fc = []
        lastidx = len(model.classifier) - 1
        for i in range(lastidx):
            fc.append(model.classifier[i])
        self.fc = nn.Sequential(*fc)
        self.features_size = model.classifier[lastidx].in_features

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
