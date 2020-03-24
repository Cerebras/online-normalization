"""
Released under BSD 3-Clause License,
Modifications are Copyright (c) 2019 Cerebras, Inc.
All rights reserved.
"""
import argparse
import os
import random
import warnings

import torch
from torch.backends import cudnn
from torchvision import transforms
from torchvision import datasets

from utils import main_worker

NUM_CLASSES = 10

parser = argparse.ArgumentParser(description='PyTorch FashionMNIST Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model-dir', type=str, default='./model_dir',
                    help='dir to which model is saved (default: ./model_dir)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.04, type=float,
                    metavar='LR', help='initial learning rate (default: 0.04)',
                    dest='lr')
parser.add_argument('--momentum', default=0, type=float, metavar='M',
                    help='optimizer momentum (default: 0)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training')
norm_choices=['batch', 'layer', 'online', 'none']
parser.add_argument('--norm-mode', default='batch', type=str,
                    metavar='NORM', choices=norm_choices,
                    help='norm choices: ' +
                        ' | '.join(norm_choices) +
                        ' (default: batch)')
parser.add_argument('--afwd', '--decay-factor-forward', default=0.999,
                    type=float, metavar='AFWD', dest='afwd',
                    help='forward decay factor which sets momentum process '
                         'hyperparameter when using online normalization '
                         '(default: 0.999)')
parser.add_argument('--abkw', '--decay-factor-backward', default=0.99,
                    type=float, metavar='ABKW', dest='abkw',
                    help='backward decay factor which sets control process '
                         'hyperparameter when using online normalization '
                         '(default: 0.99)')
ecm_choices=['ls', 'ac', 'none']
parser.add_argument('--ecm', default='ls', type=str,
                    metavar='ECM', choices=ecm_choices,
                    help='Online Norm ErrorCheckingMechanism choices: ' +
                        ' | '.join(ecm_choices) +
                        ' (default: ls)')
args = parser.parse_args()


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Data loading code
    print('=> creating training set...')
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(args.data, train=True,
                                          transform=train_transform,
                                          target_transform=None,
                                          download=True)
    print('=> create train dataloader...')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)

    print('=> creating validation set...')
    val_transform = transforms.Compose([transforms.ToTensor()])
    val_dataset = datasets.FashionMNIST(args.data, train=False,
                                        transform=val_transform,
                                        target_transform=None,
                                        download=True)
    print('=> creating validation dataloader...')
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    main_worker(train_loader, val_loader, NUM_CLASSES, args)


if __name__ == '__main__':
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
