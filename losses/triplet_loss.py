"""Triplet loss with batch-hard mining."""
from __future__ import absolute_import

import torch
from torch import nn

from utils.util import euclidean_dist


def hard_example_mining(dist_mat, labels):
    """For each anchor, find the hardest positive and hardest negative."""
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    dist_ap, _ = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    dist_an, _ = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    return dist_ap.squeeze(1), dist_an.squeeze(1)


class TripletLoss(object):
    """Batch-hard triplet loss with margin ranking loss.

    Reference: 'In Defense of the Triplet Loss for Person Re-Identification'.
    """
    def __init__(self, margin=0.3):
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def __call__(self, features, labels):
        dist_mat = euclidean_dist(features, features)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        y = dist_an.new_full((dist_an.size(0),), 1.0)
        return self.ranking_loss(dist_an, dist_ap, y)
