import torch.nn as nn
from torchvision.models import resnet50

from models.backbone.base import BaseNet


class ResNet50(BaseNet):
    def __init__(self, pretrained=True, set_eval_mode=False, **kwargs):
        super(ResNet50, self).__init__()

        model = resnet50(pretrained=pretrained)

        self.set_eval_mode = set_eval_mode
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
            nn.Flatten()
        )
        self.features_size = 2048

    def forward(self, x):
        if self.set_eval_mode:
            self.features.eval()  # to avoid updating BN stats
        x = self.features(x)
        return x
