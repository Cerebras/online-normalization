"""
Released under BSD 3-Clause License,
Modifications are Copyright (c) 2019 Cerebras, Inc.
All rights reserved.
"""
import os
import argparse
import random
import warnings

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import utils, transforms

from simulation import SimDataset, generate_random_data
from utils import main_worker


parser = argparse.ArgumentParser(description='PyTorch UNet Model Training')
parser.add_argument('--model-dir', type=str, default='./model_dir',
                    help='dir to which model is saved')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run (default: 40)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts, default: 0)')
parser.add_argument('-b', '--batch-size', default=25, type=int,
                    metavar='N',
                    help='mini-batch size (default: 25)')
parser.add_argument('--lr', '--learning-rate', default=0.04, type=float,
                    metavar='LR', help='initial learning rate (default: 0.04)',
                    dest='lr')
parser.add_argument('--lr-milestone', default=25, type=int,
                    help='epoch at which we take a learning-rate step '
                         '(default: 25)')
parser.add_argument('--lr-multiplier', default=0.1, type=float, metavar='M',
                    help='lr multiplier at lr_milestones (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='optimizer momentum (default: 0.9)')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 2e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training')
norm_choices=['batch', 'online', 'none']
parser.add_argument('--norm-mode', default='batch', type=str,
                    metavar='NORM', choices=norm_choices,
                    help='norm choices: ' +
                        ' | '.join(norm_choices) +
                        ' (default: batch)')
parser.add_argument('--afwd', '--decay-factor-forward', default=63 / 64.,
                    type=float, metavar='AFWD', dest='afwd',
                    help='forward decay factor which sets momentum process '
                         'hyperparameter when using online normalization '
                         '(default: 63 / 64)')
parser.add_argument('--abkw', '--decay-factor-backward', default=1 / 2.,
                    type=float, metavar='ABKW', dest='abkw',
                    help='backward decay factor which sets control process '
                         'hyperparameter when using online normalization '
                         '(default: 1 / 2)')
ecm_choices=['ls', 'ac', 'none']
parser.add_argument('--ecm', default='ls', type=str,
                    metavar='ECM', choices=ecm_choices,
                    help='Online Norm ErrorCompensationMechanism choices: ' +
                        ' | '.join(ecm_choices) +
                        ' (default: ls)')
parser.add_argument('--classes', type=int, default=6, metavar='N',
                    help='classes (default: 6)')
parser.add_argument('--t-size', type=int, default=2000, metavar='N',
                    help='train set size (default: 2000)')
parser.add_argument('--v-size', type=int, default=200, metavar='N',
                    help='val set size (default: 200)')
parser.add_argument('--im-size', type=int, default=192, metavar='N',
                    help='image height and width (default: 192)')
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

    # Generate some random images
    input_images, target_masks = generate_random_data(args.im_size,
                                                      args.im_size, count=3)

    print(f'=> image shape: {input_images.shape} in '
          f'range: [{input_images.min()}, {input_images.max()}]')
    print(f'=> target shape: {target_masks.shape} in '
          f'range: [{target_masks.min()}, {target_masks.max()}]')

    t_form = transforms.Compose([transforms.ToTensor(),])
    # create generator to create images
    train_set = SimDataset(args.t_size, args.im_size, transform=t_form)
    val_set = SimDataset(args.v_size, args.im_size, transform=t_form)

    # create dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # train model
    model = main_worker(train_loader, val_loader, args=args)

    return model


if __name__ == '__main__':
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
