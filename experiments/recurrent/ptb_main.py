"""
Released under BSD 3-Clause License,
Modifications are Copyright (c) 2019 Cerebras, Inc.
All rights reserved.
"""
import os
import random
import warnings
import argparse

import torch
from torch.backends import cudnn

import data
from utils import main_worker, batchify


parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model-dir', type=str, default='./model_dir',
                    help='dir to which model is saved (default: ./model_dir)')

ru_choices=['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU', 'CustomLSTM',
              'CustomRNN_tanh', 'CustomRNN_relu']
parser.add_argument('--ru-type', default='CustomLSTM', type=str,
                    metavar='RUnit', choices=ru_choices,
                    help='recurrent unit choices: ' +
                        ' | '.join(ru_choices) +
                        ' (default: CustomLSTM)')
norm_choices=['layer', 'online', 'none']
parser.add_argument('--norm-mode', default='online', type=str,
                    metavar='NORM', choices=norm_choices,
                    help='norm choices: ' +
                        ' | '.join(norm_choices) +
                        ' (default: online)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings (default: 200)')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer (default: 200)')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers  (default: 1)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run (default: 40)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts, default: 0)')
parser.add_argument('-b', '--batch-size', default=20, type=int,
                    metavar='N',
                    help='mini-batch size (default: 20)')
parser.add_argument('--bptt', type=int, default=128,
                    help='sequence length (default: 128)')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (default: 0 = no dropout)')
parser.add_argument('--not-tied', action='store_true',
                    help='do not tie the word embedding and softmax weights')
parser.add_argument('--lr', '--learning-rate', default=6.5, type=float,
                    metavar='LR', help='initial learning rate (default: 6.5)',
                    dest='lr')
parser.add_argument('--lr-decay', type=float, default=1.,
                    help='per epoch exponential learning rate schedule decay '
                         'rate of lr (default: 1, value of 1 means no decay)')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=25, type=int,
                    metavar='N', help='print frequency (default: 25)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='Perform a one time evaluation on the validation set. '
                         'Will not perform any training.')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training')
parser.add_argument('--test-at-end', action='store_true',
                    help='test at the end of training')
parser.add_argument('--afwd', '--decay-factor-forward', default=8191 / 8192,
                    type=float, metavar='AFWD', dest='afwd',
                    help='forward decay factor which sets momentum process '
                         'hyperparameter when using online normalization '
                         '(default: 8191 / 8192)')
parser.add_argument('--abkw', '--decay-factor-backward', default=31 / 32,
                    type=float, metavar='ABKW', dest='abkw',
                    help='backward decay factor which sets control process '
                         'hyperparameter when using online normalization '
                         '(default: 31 / 32)')
ecm_choices=['ls', 'ac', 'none']
parser.add_argument('--ecm', default='ls', type=str,
                    metavar='ECM', choices=ecm_choices,
                    help='Online Norm ErrorCheckingMechanism choices: ' +
                        ' | '.join(ecm_choices) +
                        ' (default: ls)')
parser.add_argument('--cell-norm', action='store_true',
                    help='normalize cell gate in LSTM')
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=> load data...')
    corpus = data.Corpus(args.data)

    eval_batch_size = 10
    train_loader = batchify(corpus.train, args.batch_size, device)
    val_loader = batchify(corpus.valid, eval_batch_size, device)

    ntokens = len(corpus.dictionary)
    main_worker(train_loader, val_loader, ntokens, args, device)


if __name__ == '__main__':
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
