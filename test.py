"""Testing protocol for IAHL-ReID on PRCC.

PRCC reports two settings: Standard (same-cloth query vs. gallery) and
Cloth-Changing (changed-cloth query vs. gallery). The PRCC gallery is the
same across both settings. This module runs the gallery once, then evaluates
each query split separately.
"""
from __future__ import absolute_import

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm


def _extract(loader, use_gpu, model):
    """Flip-test feature extraction. Each loader yields (img, gray, pid, clothid, camid).

    Only the fused feature f is used at inference (paper Sec. 3.4).
    """
    feats, pids = [], []
    model.eval()
    with torch.no_grad():
        for img, gray, pid, clothid, camid in tqdm(loader):
            flip_img = torch.flip(img, [3])
            if use_gpu:
                img, flip_img = img.cuda(), flip_img.cuda()
            f1 = model(img)['f']
            f2 = model(flip_img)['f']
            feat = f1 + f2
            feat = F.normalize(feat, p=2, dim=1)
            feats.append(feat.data.cpu())
            pids.extend(pid)
    feats = torch.cat(feats, 0)
    pids = np.asarray(pids)
    return feats, pids


def _distmat(qf, gf):
    qf = qf / (torch.norm(qf, 2, dim=-1, keepdim=True) + 1e-12)
    gf = gf / (torch.norm(gf, 2, dim=-1, keepdim=True) + 1e-12)
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return distmat.numpy()


def _eval_prcc(distmat, q_pids, g_pids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc, all_AP, num_valid_q = [], [], 0.0
    for q_idx in range(num_q):
        raw_cmc = matches[q_idx]
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0
        num_rel = raw_cmc.sum()
        tmp = raw_cmc.cumsum()
        tmp = [x / (i + 1.0) for i, x in enumerate(tmp)]
        AP = (np.asarray(tmp) * raw_cmc).sum() / num_rel
        all_AP.append(AP)
    assert num_valid_q > 0, 'all queries absent from gallery'
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return cmc, mAP


def test_prcc(args, query_sc_loader, query_cc_loader, gallery_loader, model, use_gpu,
              epoch=None):
    """Run evaluation on PRCC. Returns ((rank1_sc, mAP_sc), (rank1_cc, mAP_cc))."""
    gf, g_pids = _extract(gallery_loader, use_gpu, model)
    print('  gallery features shape:', gf.shape)

    tag = '' if epoch is None else 'Epoch {} '.format(epoch)

    qf_sc, q_pids_sc = _extract(query_sc_loader, use_gpu, model)
    distmat = _distmat(qf_sc, gf)
    cmc, mAP = _eval_prcc(distmat, q_pids_sc, g_pids)
    print('{}Standard  mAP: {:.4%}  Rank-1: {:.4%}  Rank-5: {:.4%}  Rank-10: {:.4%}'.format(
        tag, mAP, cmc[0], cmc[4], cmc[9]))

    qf_cc, q_pids_cc = _extract(query_cc_loader, use_gpu, model)
    distmat = _distmat(qf_cc, gf)
    cmc_cc, mAP_cc = _eval_prcc(distmat, q_pids_cc, g_pids)
    print('{}Clothing  mAP: {:.4%}  Rank-1: {:.4%}  Rank-5: {:.4%}  Rank-10: {:.4%}'.format(
        tag, mAP_cc, cmc_cc[0], cmc_cc[4], cmc_cc[9]))

    return (cmc[0], mAP), (cmc_cc[0], mAP_cc)
