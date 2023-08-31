import torch
import torch.nn as nn


class WCHLoss(nn.Module):
    def __init__(self, temperature=0.3, normalize_w=False):
        super().__init__()

        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.normalize_w = normalize_w
        self.losses = {}

    def forward(self, h1, h2, weighted):
        nbit = h1.size(1)
        logits = torch.einsum('ik,jk->ij', h1, h2)
        logits = (logits / nbit) / self.temperature

        balance_logits = h1.sum(0) / h1.size(0)
        reg = self.mse(balance_logits, torch.zeros_like(balance_logits)) - self.mse(h1, torch.zeros_like(h1))

        if self.normalize_w:
            ce = self.ce(logits, weighted / weighted.sum(dim=1, keepdim=True))  # scale to probability
        else:
            ce = self.ce(logits, weighted)

        loss = ce + reg
        self.losses['reg'] = reg
        self.losses['ce'] = ce

        return loss
