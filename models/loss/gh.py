import torch
import torch.nn as nn
import torch.nn.functional as F


class UnsupGHLoss(nn.Module):
    def __init__(self, alpha=1, pow=3, multiclass=False, **kwargs):
        super(UnsupGHLoss, self).__init__()

        self.alpha = alpha
        self.pow = pow
        self.multiclass = multiclass

        self.losses = {}

    def forward(self, x, h, b):
        """
        x: features before hash layer
        code_logits: output from hash FC
        labels: not using (only use to obtain size)
        onehot: not using
        """
        # case if batch data not even (normally last batch)
        if x.size(0) % 2 != 0:
            b = b[:-1]
            x = x[:-1]

        target_b = F.cosine_similarity(b[:x.size(0) // 2], b[x.size(0) // 2:])
        target_x = F.cosine_similarity(x[:x.size(0) // 2], x[x.size(0) // 2:]).detach()

        loss1 = F.mse_loss(target_b, target_x)
        # loss2 = F.mse_loss(torch.abs(h), Variable(torch.ones(h.size()).cuda()))
        loss2 = torch.mean(torch.abs(torch.pow(torch.abs(h) - 1., self.pow)))
        loss = loss1 + self.alpha * loss2

        self.losses['mse'] = loss1
        self.losses['quan'] = loss2

        return loss