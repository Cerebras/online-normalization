"""
Released under BSD 3-Clause License,
Modifications are Copyright (c) 2019 Cerebras, Inc.
All rights reserved.
"""
import os, time
import shutil

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.backends import cudnn

from model.loss import dice_loss, jeccard_sim, mAP
from model.model import UNet
from model.norm import norm as norm_layer

best_loss = 1e32


def main_worker(train_loader, val_loader, args):
    global best_loss
    
    # create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'=> device used: {device}')

    norm_kwargs = {'mode': args.norm_mode,
                   'alpha_fwd': args.afwd,
                   'alpha_bkw': args.abkw,
                   'batch_size': args.batch_size,
                   'ecm': args.ecm}

    print("=> creating model...")
    model = UNet(args.classes,
                 norm_layer=norm_layer, norm_kwargs=norm_kwargs).to(device)
    print(model)

    print("=> creating optimizer...")
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    print("=> setting up learning rate scheduler...")
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_milestone,
                                    gamma=args.lr_multiplier)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = False if args.seed else True

    if args.evaluate:
        validate(val_loader, model, args.start_epoch, device, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if epoch: scheduler.step()

        # train for one epoch
        train(train_loader, model, optimizer, epoch, device, args)

        # evaluate on validation set
        eval_loss = validate(val_loader, model, epoch, device, args)

        # remember best loss and save checkpoint
        is_best = eval_loss < best_loss
        best_loss = min(eval_loss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, args)

    print('best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model_best_file = os.path.join(args.model_dir, 'model_best.pth.tar')
    if os.path.isfile(model_best_file):
        print("=> loading checkpoint '{}'".format(model_best_file))
        checkpoint = torch.load(model_best_file)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(model_best_file, checkpoint['epoch']))

    return model


def train(train_loader, model, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    bces = AverageMeter('bces', ':.4f')
    dices = AverageMeter('dices', ':.4f')
    losses = AverageMeter('losses', ':.4f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, bces,
                             dices, losses, prefix="Train: [{}]".format(epoch))

    model.train()  # Set model to training mode

    epoch_samples = 0
    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)
        target = target.to(device)             

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        output = model(input)
        bce, dice, loss = calc_loss(output, target, phase='train')

        # record stats
        bces.update(bce.item(), input.size(0))
        dices.update(dice.item(), input.size(0))
        losses.update(loss.item(), input.size(0))

        loss.backward()
        optimizer.step()

        # statistics
        epoch_samples += input.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            progress.print(idx + 1)

    progress.print(idx + 1, 'End ')
    return losses.avg


def validate(val_loader, model, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    bces = AverageMeter('bces', ':.4f')
    dices = AverageMeter('dices', ':.4f')
    losses = AverageMeter('losses', ':.4f')
    mAPs = AverageMeter('mAPs', ':.4f')
    jeccards = AverageMeter('jeccards', ':.4f')
    progress = ProgressMeter(len(val_loader), batch_time, bces, dices, losses,
                             mAPs, jeccards, prefix="Val: [{}]".format(epoch))

    model.eval()   # Set model to evaluate mode

    epoch_samples = 0
    
    for idx, (input, target) in enumerate(val_loader):
        input = input.to(device)
        target = target.to(device)             

        # forward
        with torch.no_grad():
            output = model(input)
            bce, dice, loss, jeccard, meanAP = calc_loss(output, target,
                                                         phase='val')

            # record stats
            bces.update(bce.item(), input.size(0))
            dices.update(dice.item(), input.size(0))
            losses.update(loss.item(), input.size(0))
            mAPs.update(meanAP, input.size(0))
            jeccards.update(jeccard, input.size(0))

        # statistics
        epoch_samples += input.size(0)

        if idx % args.print_freq == 0:
            progress.print(idx + 1)

    progress.print(idx + 1, 'End ')
    return losses.avg


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


def calc_loss(pred, target, bce_weight=0.5, jec_p=.5, phase='train'):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    # calculate validation metrics as well
    if phase == 'val':
        meanAP = 0
        jeccard = 0
        for p, t in zip(pred, target):
            jeccard += jeccard_sim(p, t, jec_p=jec_p)
            
            meanAP += mAP(p, t, cutoff=jec_p)

        return bce, dice, loss, jeccard / target.size(0), meanAP / target.size(0)

    return bce, dice, loss
