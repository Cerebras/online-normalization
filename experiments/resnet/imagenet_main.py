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

from model import model as models
from utils import main_worker

IM_RESIZE_SHAPE = 256
IM_CROP_SIZE = 224
NUM_CLASSES = 1000

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model-dir', type=str, default='./model_dir',
                    help='dir to which model is saved (default: ./model_dir)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run (default: 90)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts, default: 0)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)',
                    dest='lr')
parser.add_argument('--lr-milestones', nargs='+', type=int,
                        default=[30, 60, 80, 90],
                        help='epochs at which we take a learning-rate step '
                             '(default: [30, 60, 80, 90])')
parser.add_argument('--lr-multiplier', default=0.1, type=float, metavar='M',
                    help='lr multiplier at lr_milestones (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='optimizer momentum (default: 0.9)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training')
norm_choices=['batch', 'group', 'layer', 'instance', 'online', 'none']
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
parser.add_argument('--gn-num-groups', default=32, type=int,
                    help='number of groups in group norm if using group norm '
                         'as normalization method (default: 32)')
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
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    print('=> create train dataset')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(IM_CROP_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    print('=> create train dataloader')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    print('=> create val dataloader')
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(IM_RESIZE_SHAPE),
            transforms.CenterCrop(IM_CROP_SIZE),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    main_worker(train_loader, val_loader, NUM_CLASSES, args)


if __name__ == '__main__':
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
