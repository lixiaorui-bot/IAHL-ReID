"""Training loop for IAHL-ReID on PRCC.

Multi-objective training (paper Sec. 3.4, Eqs. 9-12):

    L_total = L_reid + L_dir + lambda * L_iahl
    L_reid  = L_id(f)  + L_tri(f)     (fused retrieval feature)
    L_dir   = L_id(fd) + L_tri(fd)    (DIR Branch feature)
    L_iahl  = L_id(fh)                (IAHL Branch, auxiliary identity loss)

No mask, no skeleton, no external pseudo-label.
"""
from __future__ import absolute_import

from tqdm import tqdm

import torch

from utils.util import AverageMeter


def train_prcc(args, epoch, train_loader, model, classifiers, optimizer,
               class_criterion, metric_criterion, use_gpu):
    """One-epoch training on PRCC.

    Each batch yields (img, gray, pid, clothid, camid).
    `classifiers` is a dict with an identity classifier for each of the
    fused feature f, the DIR feature fd and the IAHL feature fh.
    """
    model.train()
    for c in classifiers.values():
        c.train()
    cls_meter = AverageMeter()
    tri_meter = AverageMeter()
    dir_cls_meter = AverageMeter()
    dir_tri_meter = AverageMeter()
    iahl_cls_meter = AverageMeter()

    for batch_idx, (img, gray, pid, clothid, camid) in enumerate(tqdm(train_loader)):
        if use_gpu:
            img = img.cuda()
            pid = pid.cuda()

        feats = model(img)

        # L_reid: identity classification + triplet on the fused feature (Eq. 9)
        y = classifiers['f'](feats['f'])
        cls_loss = class_criterion(y, pid)
        tri_loss = metric_criterion(feats['f'], pid)
        reid_loss = cls_loss + tri_loss

        # L_dir: identity classification + triplet on the DIR feature (Eq. 10)
        y_dir = classifiers['fd'](feats['fd'])
        dir_cls_loss = class_criterion(y_dir, pid)
        dir_tri_loss = metric_criterion(feats['fd'], pid)
        dir_loss = dir_cls_loss + dir_tri_loss

        # L_iahl: auxiliary identity classification on the IAHL feature (Eq. 11)
        y_iahl = classifiers['fh'](feats['fh'])
        iahl_cls_loss = class_criterion(y_iahl, pid)
        iahl_loss = iahl_cls_loss

        # L_total (Eq. 12)
        loss = reid_loss + dir_loss + args.iahl_loss_weight * iahl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cls_meter.update(cls_loss.item(), pid.size(0))
        tri_meter.update(tri_loss.item(), pid.size(0))
        dir_cls_meter.update(dir_cls_loss.item(), pid.size(0))
        dir_tri_meter.update(dir_tri_loss.item(), pid.size(0))
        iahl_cls_meter.update(iahl_cls_loss.item(), pid.size(0))

    if args.print_train_info_epoch_freq != -1 and epoch % args.print_train_info_epoch_freq == 0:
        print('Epoch{:<3} ReID Cls:{:.4f} Tri:{:.4f} | DIR Cls:{:.4f} Tri:{:.4f} | IAHL Cls:{:.4f}'.format(
            epoch, cls_meter.avg, tri_meter.avg,
            dir_cls_meter.avg, dir_tri_meter.avg, iahl_cls_meter.avg))
