import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.vit import HuggingFaceViT


# https://arxiv.org/abs/2209.14099
# https://github.com/RosieYuu/WCH

class WCH(nn.Module):
    def __init__(self,
                 transformer: HuggingFaceViT,
                 nbit: int,
                 tau_w: float = 0.2,
                 **kwargs):
        super(WCH, self).__init__()
        self.nbit = nbit

        self.transformer = transformer
        self.fc = nn.Linear(transformer.features_size, transformer.features_size)
        self.b = nn.Linear(transformer.features_size, nbit)
        self.fc_norm = nn.LayerNorm(transformer.features_size, eps=1e-6)
        self.tau_w = tau_w
        self._init_weights()

    def get_backbone(self):
        return self.transformer

    def get_training_modules(self):
        return nn.ModuleDict({
            'fc': self.fc,
            'fc_norm': self.fc_norm,
            'b': self.b
        })

    def hash_head(self, x):
        x = F.gelu(self.fc(x))
        x = torch.tanh(self.b(x))
        h = x.sign()
        return x + torch.detach(h - x)

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fc.weight, mode='fan_out')
        nn.init.normal_(self.fc.bias)
        nn.init.kaiming_uniform_(self.b.weight, mode='fan_out')
        nn.init.normal_(self.b.bias)

    def cross_attention(self, x1, x2, T=1):
        sim = torch.einsum('nid,njd->nij', F.normalize(x1, dim=-1), F.normalize(x2, dim=-1))
        sim1 = torch.softmax(sim.permute(0, 2, 1) / T, dim=-1)
        sim2 = torch.softmax(sim / T, dim=-1)
        x1 = torch.einsum('nid,ndj->nij', sim1, x1)
        x2 = torch.einsum('nid,ndj->nij', sim2, x2)
        return x1, x2

    def weighted(self, x1, x2, T=0.2):
        sim = torch.einsum('nid,mjd->nmij', F.normalize(x1, dim=-1), F.normalize(x2, dim=-1))
        sim = sim.max(-1)[0].mean(-1)
        sim = torch.softmax(sim / T, dim=-1)

        scale = torch.eye(sim.size(0), device=sim.device)
        scale = torch.einsum('ni,ni->n', scale, sim)

        weighted = sim / scale
        return weighted

    def train_forward(self, img1, img2):
        x1 = self.transformer.model(img1)[0]
        x2 = self.transformer.model(img2)[0]
        x1 = x1[:, 1:, :]
        x2 = x2[:, 1:, :]

        # x1 = self.patch_fc(x1)
        # x2 = self.patch_fc(x2)

        x1, x2 = self.cross_attention(x1, x2)
        weighted = self.weighted(x1, x2, self.tau_w)

        x1 = self.fc_norm(x1.mean(dim=1))
        x2 = self.fc_norm(x2.mean(dim=1))
        h1 = self.hash_head(x1)
        h2 = self.hash_head(x2)

        return h1, h2, weighted

    def forward(self, x):
        x = self.transformer.model(x)[0]
        x = x[:, 1:, :].mean(dim=1)
        x = self.fc_norm(x)
        logits = self.hash_head(x)
        return logits
