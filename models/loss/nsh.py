import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.softsort import SoftSort


class NSHLoss(nn.Module):

    def __init__(self,
                 m=2,
                 quan=1,
                 tau: float = 0.1,
                 softsort_tau: float = 64.,
                 **kwargs):
        super(NSHLoss, self).__init__()

        self.m = m
        self.quan = quan
        self.tau = tau
        self.softsort = SoftSort(softsort_tau)
        self.losses = {}

    def forward(self, b, z):
        """
        x: features before hash layer
        v: output from hash FC
        labels: not using (only use to obtain size)
        """
        b1, b2 = b[:b.size(0) // 2], b[b.size(0) // 2:]
        z1, z2 = z[:z.size(0) // 2], z[z.size(0) // 2:]
        s = (b1 @ b2.t()) / (2 * b.size(1)) + 0.5
        p = self.softsort(s)
        # print(p[0])
        e = torch.einsum('ijk,kd->ijd', p, z1)
        cossim = torch.einsum('ijd,id->ij', e, z2)

        labels = torch.zeros(b1.size(0), device=b1.device).long()
        ce_loss = torch.tensor(0., device=b1.device)

        for i in range(self.m):
            pos, neg = cossim[:, i:i + 1], cossim[:, self.m:]
            logits = torch.softmax(torch.cat([pos, neg], dim=1) / self.tau, dim=-1)
            ce_loss += F.cross_entropy(logits, labels) / self.m

        quan_loss = F.mse_loss(b, b.sign())
        loss = ce_loss + self.quan * quan_loss

        self.losses['nce'] = ce_loss
        self.losses['quan'] = quan_loss
        return loss
