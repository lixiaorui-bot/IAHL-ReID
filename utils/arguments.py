"""CLI arguments for IAHL-ReID on PRCC."""
from __future__ import absolute_import
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # ---------- Data ----------
    parser.add_argument('--dataset_root', type=str, default='data',
                        help='root path to PRCC data directory')
    parser.add_argument('--dataset_filename', type=str, default='rgb',
                        help='sub-folder under dataset_root (default: rgb)')
    parser.add_argument('--height', type=int, default=384, help='input image height')
    parser.add_argument('--width', type=int, default=192, help='input image width')
    parser.add_argument('--horizontal_flip_pro', type=float, default=0.5,
                        help='random horizontal flip probability')
    parser.add_argument('--pad_size', type=int, default=10, help='padding size')
    parser.add_argument('--random_erasing_pro', type=float, default=0.5,
                        help='random erasing probability')
    parser.add_argument('--train_batch', default=32, type=int, help='train batch size')
    parser.add_argument('--test_batch', default=128, type=int, help='test batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='data loader workers')
    parser.add_argument('--num_instances', type=int, default=8,
                        help='number of instances per identity in a batch')

    # ---------- Optimization ----------
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--max_epoch', default=80, type=int, help='maximum training epoch')
    parser.add_argument('--margin', type=float, default=0.3, help='triplet margin')
    parser.add_argument('--lr', default=0.00035, type=float, help='initial learning rate')
    parser.add_argument('--warm_up_factor', default=0.01, type=float, help='warm-up factor')
    parser.add_argument('--warm_up_epochs', default=10, type=int, help='warm-up epochs')
    parser.add_argument('--step_milestones', default=[20, 40, 60], nargs='*', type=int,
                        help='epoch milestones for LR decay')
    parser.add_argument('--gamma', default=0.1, type=float, help='LR decay ratio')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')

    # ---------- Hardware / IO ----------
    parser.add_argument('--resume', type=str, default='', help='resume from a checkpoint')
    parser.add_argument('--gpu_devices', default='0', type=str,
                        help='gpu ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--use_cpu', action='store_true', help='force CPU')
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--evaluate', action='store_true', help='evaluation only')
    parser.add_argument('--print_train_info_epoch_freq', type=int, default=5,
                        help='print training info every N epoch (-1 disables)')
    parser.add_argument('--start_eval_epoch', type=int, default=0,
                        help='start to evaluate after this epoch')
    parser.add_argument('--eval_epoch', type=int, default=5,
                        help='run evaluation every N epochs (set <=0 to test only at end)')
    parser.add_argument('--save_dir', type=str, default='log', help='output directory')
    parser.add_argument('--save_checkpoint', action='store_true', help='save model checkpoint')

    # ---------- IAHL-specific (paper Sec. 3 & Sec. 4.2) ----------
    parser.add_argument('--num_parts', type=int, default=4,
                        help='K: number of horizontal strips in the DIR Branch')
    parser.add_argument('--iahl_loss_weight', type=float, default=0.3,
                        help='lambda: weight of the auxiliary identity loss L_iahl')

    return parser.parse_args()
