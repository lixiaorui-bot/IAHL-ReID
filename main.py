"""IAHL-ReID entry point — train / evaluate on PRCC.

RGB-only cloth-changing person Re-ID (paper Sec. 3):
ResNet-50 backbone with a DIR Branch (global + K horizontal local features)
and an independently supervised IAHL Branch (CSIG), fused into the final
retrieval feature. Trained with L_total = L_reid + L_dir + lambda * L_iahl.
No external masks; uses only RGB images on the PRCC dataset.
"""
from __future__ import absolute_import

import datetime
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
from torch import nn

from data_process.dataset_loader_cc import get_prcc_dataset_loader
from dataset import PRCC
from losses.cross_entropy_loss import CrossEntropyLabelSmooth
from losses.triplet_loss import TripletLoss
from model.iahl import IAHL, Classifier
from scheduler.warm_up_multi_step_lr import WarmupMultiStepLR
from test import test_prcc
from train import train_prcc
from utils.arguments import get_args
from utils.util import Logger, load_checkpoint, save_checkpoint, set_random_seed


def main():
    args = get_args()
    if not osp.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    sys.stdout = Logger(osp.join(args.save_dir, 'log.txt'))
    sys.stderr = sys.stdout

    for k, v in sorted(vars(args).items()):
        print('%s: %s' % (k, v))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available() and not args.use_cpu

    if use_gpu:
        print('Currently using GPU', args.gpu_devices)
    else:
        print('Currently using CPU')
    set_random_seed(args.seed, use_gpu)

    print('Initializing dataset PRCC ...')
    dataset = PRCC.PRCC(dataset_root=args.dataset_root, dataset_filename=args.dataset_filename)
    train_loader, query_sc_loader, query_cc_loader, gallery_loader = \
        get_prcc_dataset_loader(dataset, args=args, use_gpu=use_gpu)

    num_classes = dataset.num_train_pids
    model = IAHL(last_stride=1,
                 pretrain=True,
                 num_classes=num_classes,
                 num_parts=args.num_parts)

    # One identity classifier per supervised feature: f (fused), fd (DIR), fh (IAHL)
    classifiers = nn.ModuleDict({
        'f': Classifier(feature_dim=model.feature_dim, num_classes=num_classes),
        'fd': Classifier(feature_dim=model.feature_dim, num_classes=num_classes),
        'fh': Classifier(feature_dim=model.feature_dim, num_classes=num_classes),
    })

    class_criterion = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=0.1, use_gpu=use_gpu)
    metric_criterion = TripletLoss(margin=args.margin)

    parameters = list(model.parameters()) + list(classifiers.parameters())
    optimizer = torch.optim.Adam(params=[{'params': parameters, 'initial_lr': args.lr}],
                                 lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = args.start_epoch
    if args.resume:
        print('Loading checkpoint from', args.resume)
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        classifiers.load_state_dict(checkpoint['classifiers_state_dict'], strict=True)
        if 'optimizer_state_dict' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if use_gpu:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
            start_epoch = checkpoint['epoch'] + 1

    scheduler = WarmupMultiStepLR(optimizer,
                                  milestones=args.step_milestones,
                                  gamma=args.gamma,
                                  warmup_factor=args.warm_up_factor,
                                  warmup_iters=args.warm_up_epochs,
                                  last_epoch=start_epoch - 1)

    if use_gpu:
        model = nn.DataParallel(model).cuda()
        classifiers = classifiers.cuda()

    if args.evaluate:
        print('Evaluate only')
        test_prcc(args, query_sc_loader, query_cc_loader, gallery_loader, model, use_gpu, epoch=None)
        return 0

    print('==> Start training')
    start_time = time.time()
    train_time = 0
    best_mAP = -np.inf
    best_rank1 = -np.inf
    best_epoch_mAP = 0
    best_epoch_rank1 = 0
    best_mAP_cc = -np.inf
    best_rank1_cc = -np.inf
    best_epoch_mAP_cc = 0
    best_epoch_rank1_cc = 0

    eval_every = args.eval_epoch if args.eval_epoch > 0 else args.max_epoch + 1
    for epoch in range(start_epoch, args.max_epoch):
        t0 = time.time()
        train_prcc(args, epoch + 1, train_loader, model, classifiers, optimizer,
                   class_criterion, metric_criterion, use_gpu)
        train_time += round(time.time() - t0)
        scheduler.step()  # paper: 10-epoch linear warmup, x0.1 decay at epochs 20/40/60

        if (epoch + 1) > args.start_eval_epoch and (epoch + 1) % eval_every == 0 \
                or (epoch + 1) == args.max_epoch:
            print('==> Test (epoch {})'.format(epoch + 1))
            (rank1_sc, mAP_sc), (rank1_cc, mAP_cc) = test_prcc(
                args, query_sc_loader, query_cc_loader, gallery_loader,
                model, use_gpu, epoch=epoch + 1)
            flag = True

            if mAP_sc > best_mAP:
                best_mAP = mAP_sc
                best_epoch_mAP = epoch + 1
            if rank1_sc > best_rank1:
                best_rank1 = rank1_sc
                best_epoch_rank1 = epoch + 1
            if mAP_cc > best_mAP_cc:
                best_mAP_cc = mAP_cc
                best_epoch_mAP_cc = epoch + 1
            if rank1_cc > best_rank1_cc:
                best_rank1_cc = rank1_cc
                best_epoch_rank1_cc = epoch + 1

            if args.save_checkpoint:
                save_checkpoint({
                    'model_state_dict': model.module.state_dict() if use_gpu else model.state_dict(),
                    'classifiers_state_dict': classifiers.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rank1': rank1_sc,
                    'mAP': mAP_sc,
                    'epoch': epoch,
                }, mAP_sc > best_mAP, rank1_sc > best_rank1,
                osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1)
                         + '_mAP_' + str(round(mAP_sc * 100, 2))
                         + '_rank1_' + str(round(rank1_sc * 100, 2)) + '.pth'))

    print('==> Best Standard mAP {:.4%}, epoch {}'.format(best_mAP, best_epoch_mAP))
    print('==> Best Standard Rank-1 {:.4%}, epoch {}'.format(best_rank1, best_epoch_rank1))
    print('==> Best Cloth-Changing mAP {:.4%}, epoch {}'.format(best_mAP_cc, best_epoch_mAP_cc))
    print('==> Best Cloth-Changing Rank-1 {:.4%}, epoch {}'.format(best_rank1_cc, best_epoch_rank1_cc))
    elapsed = str(datetime.timedelta(seconds=round(time.time() - start_time)))
    train_time = str(datetime.timedelta(seconds=train_time))
    print('Finished. Total {} (training {}).'.format(elapsed, train_time))


if __name__ == '__main__':
    main()
