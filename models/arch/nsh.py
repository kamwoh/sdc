import torch.nn as nn

from models.arch.base import BaseNet
from models.layers.l2norm import L2Norm
from models.layers.softsort import SoftSort


# https://arxiv.org/abs/2201.13322

class NSH(BaseNet):

    def __init__(self,
                 backbone: nn.Module,
                 nbit: int,
                 nclass: int,
                 latent_dim: int = 1024,
                 **kwargs):
        super().__init__(backbone, nbit, nclass, **kwargs)

        self.softsort = SoftSort(nbit)

        self.hash_encoder = nn.Sequential(
            nn.Linear(self.backbone.features_size, nbit),
            # nn.BatchNorm1d(nbit),
            nn.Tanh()
        )
        self.latent_encoder = nn.Sequential(
            nn.Linear(self.backbone.features_size, latent_dim),
            L2Norm(dim=-1)
        )

    def get_training_modules(self):
        return nn.ModuleDict({'hash_encoder': self.hash_encoder,
                              'latent_encoder': self.latent_encoder})

    def forward(self, x):
        x = self.backbone(x)
        b = self.hash_encoder(x)
        l = self.latent_encoder(x)
        return x, b, l
