"""
Released under BSD 3-Clause License,
Modifications are Copyright (c) 2019 Cerebras, Inc.
All rights reserved.
"""
import os
import sys
import time
import math
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR

import model.model as models

best_ppl = 1e32


def main_worker(train_loader, val_loader, ntokens, args, device):
    global best_ppl

    model_kwargs = {'dropout': args.dropout,
                    'tie_weights': not args.not_tied,
                    'norm': args.norm_mode,
                    'alpha_fwd': args.afwd,
                    'alpha_bkw': args.abkw,
                    'batch_size': args.batch_size,
                    'ecm': args.ecm,
                    'cell_norm': args.cell_norm }

    # create model
    print("=> creating model: '{}'".format(args.ru_type))
    model = models.RNNModel(args.ru_type, ntokens, args.emsize, args.nhid,
                            args.nlayers, **model_kwargs).to(device)

    print(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                weight_decay=args.weight_decay)

    scheduler = ExponentialLR(optimizer, gamma=1 / args.lr_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_ppl = checkpoint['best_ppl']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = False if args.seed else True

    if args.evaluate:
        validate(val_loader, model, criterion, device, args, ntokens)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if epoch: scheduler.step()

        # train for one epoch
        train(train_loader, model,
                    criterion, optimizer, epoch, device, args, ntokens)

        # evaluate on validation set
        ppl = validate(val_loader, model, criterion, device, args, ntokens)

        # remember best ppl and save checkpoint
        is_best = ppl < best_ppl
        best_ppl = min(ppl, best_ppl)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_ppl': best_ppl,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, args)


def train(train_loader, model,
          criterion, optimizer, epoch, device, args, ntokens):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    ppl = AverageMeter('ppl', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses,
                             ppl, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    hidden = model.init_hidden(args.batch_size)

    end = time.time()
    batch, idx = 0, 0
    while idx < len(train_loader) - 1:
        input, target, seq_len = get_batch(train_loader, idx, args)
        # measure data loading time
        data_time.update(time.time() - end)

        # Starting each batch, we detach the hidden state from how it was
        # previously produced.
        hidden = repackage_hidden(hidden)

        # compute output
        output, hidden = model(input, hidden)
        loss = criterion(output.view(-1, ntokens), target)

        # record stats
        losses.update(loss.item(), input.size(0))
        ppl.update(math.exp(loss), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch % args.print_freq == 0:
            progress.print(idx + 1)

        idx += seq_len; batch += 1

    progress.print(idx + 1, 'End ')
    return ppl.avg

def validate(val_loader, model, criterion, device, args, ntokens):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    ppl = AverageMeter('ppl', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, ppl,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    hidden = model.init_hidden(val_loader.size(1))

    with torch.no_grad():
        end = time.time()
        batch, idx = 0, 0
        while idx < len(val_loader) - 1:
            input, target, seq_len = get_batch(val_loader,
                                               idx, args, train=False)

            # Starting each batch, we detach the hidden state from how it was
            # previously produced.
            hidden = repackage_hidden(hidden)

            # compute output
            output, hidden = model(input, hidden)
            loss = criterion(output.view(-1, ntokens), target)

            # record stats
            losses.update(loss.item(), input.size(0))
            ppl.update(math.exp(loss), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch % args.print_freq == 0:
                progress.print(idx + 1)

            idx += seq_len; batch += 1

    progress.print(idx + 1, 'End ')
    return ppl.avg


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    model_fname = os.path.join(args.model_dir, filename)
    torch.save(state, model_fname)
    if is_best:
        shutil.copyfile(model_fname,
                        os.path.join(args.model_dir, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch, prepend=''):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print(prepend + '\t'.join(entries))


    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def batchify(data, bsz, device):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' can not be learned, but allows more
    efficient batch processing.
    """
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    If we didn't, the rnn_model would try backpropagating all the way to start
    of the dataset
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return list(repackage_hidden(v) for v in h)


def get_batch(source, i, args, train=True):
    """
    get_batch subdivides the source data into chunks of length args.bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM.
    """
    if train:
        seq_len = random.randint(2, args.bptt)
        seq_len = min(seq_len, len(source) - 1 - i)
    else:
        seq_len = min(32, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target, seq_len

