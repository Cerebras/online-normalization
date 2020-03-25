"""
Released under BSD 3-Clause License,
Modifications are Copyright (c) 2019 Cerebras, Inc.
All rights reserved.
"""
import os
import sys
import time
import shutil

import torch
from torch import nn
from torch.backends import cudnn
from torch.optim.lr_scheduler import MultiStepLR

from model import model as models
from model.norm import norm as norm_layer

best_acc1 = 0


def main_worker(train_loader, val_loader, num_classes, args, cifar=False):
    global best_acc1

    scale_lr_and_momentum(args, cifar=cifar)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    norm_kwargs = {'mode': args.norm_mode,
                   'alpha_fwd': args.afwd,
                   'alpha_bkw': args.abkw,
                   'batch_size': args.batch_size,
                   'ecm': args.ecm,
                   'gn_num_groups': args.gn_num_groups}
    model_kwargs = {'num_classes': num_classes,
                    'norm_layer': norm_layer,
                    'norm_kwargs': norm_kwargs,
                    'cifar': cifar,
                    'kernel_size': 3 if cifar else 7,
                    'stride': 1 if cifar else 2,
                    'padding': 1 if cifar else 3,
                    'inplanes': 16 if cifar else 64}
    if cifar:
        model_kwargs['depth'] = args.depth
        args.arch = 'resnetD'

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True,
                                           **model_kwargs).to(device)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](**model_kwargs).to(device)

    print(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(get_parameter_groups(model, cifar=cifar),
                                args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = MultiStepLR(optimizer,
                            milestones=args.lr_milestones,
                            gamma=args.lr_multiplier)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = False if args.seed else True

    if args.evaluate:
        validate(val_loader, model, criterion, device, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if epoch: scheduler.step()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, device, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, args)


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses,
                             top1, top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)
        target = target.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5, = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)


def validate(val_loader, model, criterion, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5, = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


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

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values
    of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_parameter_groups(model, norm_weight_decay=0, cifar=False):
    """
    Separate model parameters from scale and bias parameters following norm if
    training imagenet
    """
    if cifar:
        return model.parameters()

    model_params = []
    norm_params = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            if 'fc' not in name and ('norm' in name or 'bias' in name):
                norm_params += [p]
            else:
                model_params += [p]

    return [{'params': model_params},
            {'params': norm_params,
             'weight_decay': norm_weight_decay}]


def scale_lr_and_momentum(args, cifar=False, skip=False):
    """
    Scale hyperparameters given the adjusted batch_size from input
    hyperparameters and batch size

    Arguements:
        args: holds the script arguments
        cifar: boolean if we are training imagenet or cifar
        skip: boolean skipping the hyperparameter scaling.

    """
    if skip:
        return args

    print('=> adjusting learning rate and momentum. '
          f'Original lr: {args.lr}, Original momentum: {args.momentum}')

    std_b_size = 128 if cifar else 256
    
    old_momentum = args.momentum
    args.momentum = old_momentum ** (args.batch_size / std_b_size)
    args.lr = args.lr * (args.batch_size / std_b_size *
                         (1 - args.momentum) / (1 - old_momentum))

    print(f'lr adjusted to: {args.lr}, momentum adjusted to: {args.momentum}')

    return args

