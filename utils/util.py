"""Generic utilities for IAHL-ReID.

Only functions still used by the PRCC training/testing code remain: random
seed, image IO, checkpoint IO, console/file logger, average meter, distance
matrix used by the triplet loss.
"""
from __future__ import absolute_import

import errno
import os
import os.path as osp
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image


def set_random_seed(seed, use_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def read_image(img_path):
    """Read image with retry — avoids transient IOError under heavy IO load."""
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will retry.".format(img_path))
    return img


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    """Mirror stdout/stderr to a file."""
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'a')

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def load_checkpoint(fpath):
    if not osp.isfile(fpath):
        raise ValueError("no checkpoint at '{}'".format(fpath))
    checkpoint = torch.load(fpath, weights_only=False)
    print("=> Loaded checkpoint '{}'".format(fpath))
    return checkpoint


def save_checkpoint(state, is_best_mAP, is_best_rank1, fpath='checkpoint.pth'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best_mAP:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_mAP_model.pth'))
    if is_best_rank1:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_rank1_model.pth'))


def euclidean_dist(x, y):
    """L2 pairwise distance matrix. Numerical stability via clamp before sqrt."""
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist.clamp(min=1e-12).sqrt()
