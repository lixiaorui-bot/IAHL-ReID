"""IAHL-ReID model — Identity-Aware Heatmap Learning for PRCC.

Aligned with the paper (Section 3):

- Shared backbone: ImageNet-pretrained ResNet-50, shared feature map F from the
  last convolutional stage.
- DIR Branch (Sec. 3.3): global average pooling produces f_g; F is horizontally
  split into K non-overlapping strips and locally average-pooled into
  {f_k}; fd = phi_d(Concat(f_g, f_1..f_K)).
- IAHL Branch / CSIG (Sec. 3.2): z_id = phi_id(GAP(F));
  a_c = sigmoid(MLP(z_id)); F_c = a_c * F; A_s = sigmoid(Conv1x1(F_c));
  F_h = A_s * F_c; f_h = phi_h(GAP(F_h)).
- Feature fusion (Sec. 3.4): f = phi_f(Concat(fd, f_h)). During inference only
  the fused feature f is used for retrieval.

Training losses live in train.py: L_total = L_reid(f) + L_dir(fd) +
lambda * L_iahl(f_h) with lambda = 0.3 (paper Eqs. 9-12).
"""
from __future__ import absolute_import

import torch
import torch.nn as nn
import torchvision


def _weights_init_kaiming(m):
    name = m.__class__.__name__
    if 'Linear' in name:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif 'Conv' in name:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif 'BatchNorm' in name:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def _weights_init_classifier(m):
    name = m.__class__.__name__
    if 'Linear' in name:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class Classifier(nn.Module):
    """Identity classifier: feature -> num_classes logits."""
    def __init__(self, feature_dim=2048, num_classes=-1):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)
        self.classifier.apply(_weights_init_classifier)

    def forward(self, x):
        return self.classifier(x)


class IAHL(nn.Module):
    """IAHL-ReID model (paper Sec. 3).

    Args:
        last_stride: stride of backbone.layer4 (1 keeps the largest spatial map)
        pretrain: load ImageNet pretrained ResNet-50
        num_classes: number of training identities (unused here; classifiers
            are constructed separately in main.py)
        num_parts: K, number of horizontal strips in the DIR Branch
        se_reduction: reduction ratio of the channel-response MLP in CSIG
    """
    def __init__(self, last_stride=1, pretrain=True, num_classes=None,
                 num_parts=4, se_reduction=16):
        super().__init__()
        self.num_classes = num_classes
        self.num_parts = num_parts
        self.feature_dim = 2048  # dim of fd, fh and the fused feature f
        channels = self.feature_dim

        resnet = torchvision.models.resnet50(pretrained=pretrain)
        resnet.layer4[0].downsample[0].stride = (last_stride, last_stride)
        resnet.layer4[0].conv2.stride = (last_stride, last_stride)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ----- DIR Branch (Sec. 3.3) -----
        # phi_d: projection head over Concat(f_g, f_1..f_K), dim (K+1)*C -> C
        self.dir_projection = nn.Sequential(
            nn.Linear(channels * (num_parts + 1), channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.dir_projection.apply(_weights_init_kaiming)

        # ----- IAHL Branch / CSIG (Sec. 3.2) -----
        # phi_id: identity projection head producing the descriptor z_id
        self.identity_projection = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.identity_projection.apply(_weights_init_kaiming)
        # channel-wise identity responses a_c = sigma(MLP(z_id))
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels // se_reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // se_reduction, channels),
            nn.Sigmoid(),
        )
        self.channel_mlp.apply(_weights_init_kaiming)
        # spatial identity heatmap A_s = sigma(Conv1x1(F_c))
        self.spatial_conv = nn.Conv2d(channels, 1, 1)
        nn.init.kaiming_normal_(self.spatial_conv.weight, a=0, mode='fan_in')
        nn.init.constant_(self.spatial_conv.bias, 0.0)
        # phi_h: projection head producing the supplementary representation f_h
        self.iahl_projection = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.iahl_projection.apply(_weights_init_kaiming)

        # ----- Feature fusion (Sec. 3.4) -----
        # phi_f: fusion head over Concat(fd, f_h), dim 2C -> C
        self.fusion_projection = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.fusion_projection.apply(_weights_init_kaiming)

    def forward(self, x_rgb):
        """Returns a dict with the fused feature f and the branch features
        fd (DIR) and fh (IAHL). Only f is used at inference (paper Sec. 3.4).
        """
        fmap = self.backbone(x_rgb)  # shared feature map F, shape (B, C, H, W)

        # ----- DIR Branch: global + K local horizontally-pooled features -----
        fg = self.gap(fmap).flatten(1)                              # (B, C)
        height = fmap.size(2)
        part_feats = []
        for k in range(self.num_parts):
            top = height * k // self.num_parts
            bottom = height * (k + 1) // self.num_parts
            fk = self.gap(fmap[:, :, top:bottom, :]).flatten(1)    # (B, C)
            part_feats.append(fk)
        fd = self.dir_projection(torch.cat([fg] + part_feats, dim=1))

        # ----- IAHL Branch / CSIG -----
        zid = self.identity_projection(self.gap(fmap).flatten(1))  # phi_id(GAP(F))
        ac = self.channel_mlp(zid).unsqueeze(-1).unsqueeze(-1)     # a_c, (B, C, 1, 1)
        fc = fmap * ac                                             # channel recalibration
        a_s = torch.sigmoid(self.spatial_conv(fc))                 # spatial heatmap, (B, 1, H, W)
        fh_map = a_s * fc                                          # heatmap-guided feature map F_h
        fh = self.iahl_projection(self.gap(fh_map).flatten(1))     # phi_h(GAP(F_h))

        # ----- Feature fusion -----
        f = self.fusion_projection(torch.cat([fd, fh], dim=1))

        return {'f': f, 'fd': fd, 'fh': fh}
