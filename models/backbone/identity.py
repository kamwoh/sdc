import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, features_size, *args, **kwargs):
        super(Identity, self).__init__()

        self.features_size = features_size

    def forward(self, x):
        return x
