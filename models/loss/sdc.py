import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


class SDCLoss(nn.Module):

    def __init__(self,
                 rec=1,
                 quan=0,
                 quan_type='cs',
                 beta_ab=2,
                 beta_a=-1,
                 beta_b=-1,
                 ortho_constraint=True,
                 **kwargs):
        super(SDCLoss, self).__init__()

        ##### loss scale #####
        self.rec = rec
        self.quan = quan
        self.quan_type = quan_type

        self.beta_ab = beta_ab
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.ortho_constraint = ortho_constraint
        self.losses = {}

    def _one_way_distance_compute(self, a, b):
        hash_dist = F.cosine_similarity(a, b.detach())
        return hash_dist

    def find_pairs(self, x, mode='simple'):
        """

        :param x:
        :return: index
        """
        a_idx = torch.arange(0, x.size(0) // 2).to(x.device)
        b_idx = torch.arange(x.size(0) // 2, x.size(0)).to(x.device)

        assert a_idx.size(0) == b_idx.size(0), 'a and b must be equal size'

        return a_idx, b_idx

    def distance_preserving_loss(self, orig_x, hash_x):
        ##### calculate cosine similarity between pairs #####
        a_idx, b_idx = self.find_pairs(orig_x, 'simple')
        a, b = hash_x[a_idx], hash_x[b_idx]
        hash_dist_ab = self._one_way_distance_compute(a, b)
        hash_dist_ba = self._one_way_distance_compute(b, a)
        orig_dist = F.cosine_similarity(orig_x[a_idx], orig_x[b_idx]).detach()

        ##### normalize distance #####
        beta_a = self.beta_a
        beta_b = self.beta_b
        if beta_a == -1 and beta_b == -1:
            beta_a = beta_b = self.beta_ab
        elif beta_a == -1 or beta_b == -1:
            raise ValueError('set beta_a AND beta_b to have non negative values')

        orig_dist, sortidx = torch.sort(orig_dist)
        hash_dist_ab = hash_dist_ab[sortidx]  # sort based on prior
        hash_dist_ba = hash_dist_ba[sortidx]  # sort based on prior

        batch_size = orig_dist.size(0)
        grid = np.arange(1., 2 * batch_size, 2.).astype(np.float32) / (2 * batch_size)
        rand_dist = stats.beta.ppf(grid, a=beta_a, b=beta_b)  # cmf
        rand_dist = torch.tensor(rand_dist * 2 - 1).to(orig_dist.device).float()

        if self.ortho_constraint:
            rand_dist = rand_dist.relu()

        ##### distance reconstruction loss #####
        reduction = 'mean'

        # loss_rec_ab = F.l1_loss(hash_dist_ab, rand_dist, reduction=reduction)
        loss_rec_ab = (hash_dist_ab - rand_dist).relu().mean()
        # loss_rec_ba = F.l1_loss(hash_dist_ba, rand_dist, reduction=reduction)
        loss_rec_ba = (hash_dist_ba - rand_dist).relu().mean()

        loss_rec = 0.5 * loss_rec_ab + 0.5 * loss_rec_ba

        return loss_rec

    def quantization_loss(self, v):
        if self.quan_type == 'cs':
            loss_quan = (1 - torch.cosine_similarity(v, v.sign()))
        elif self.quan_type == 'l1':
            loss_quan = torch.abs(v - v.sign())
        elif self.quan_type == 'l2':
            loss_quan = torch.square(v - v.sign())
        else:  # p3, greedyhash method
            loss_quan = torch.abs(torch.pow(v - v.sign(), 3))

        loss_quan = loss_quan.mean()
        return loss_quan

    def forward(self, x, v):
        """
        x: features before hash layer
        v: output from hash FC
        labels: not using (only use to obtain size)
        """
        # case if batch data not even (normally last batch)
        if x.size(0) % 2 != 0:
            v = v[:-1]
            x = x[:-1]

        loss = torch.tensor(0.).to(x.device)

        ##### angular reconstructive embedding #####
        loss_rec = self.distance_preserving_loss(x, v)

        self.losses['rec'] = loss_rec
        loss = loss + self.rec * loss_rec

        ##### quantization error #####
        loss_quan = self.quantization_loss(v)
        self.losses['quan'] = loss_quan
        loss = loss + self.quan * loss_quan

        return loss
