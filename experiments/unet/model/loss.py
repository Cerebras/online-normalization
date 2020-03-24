"""
Released under BSD 3-Clause License,
Modifications are Copyright (c) 2019 Cerebras, Inc.
All rights reserved.
"""
import torch
import torch.nn as nn
from torch import finfo

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +
                                                 target.sum(dim=2).sum(dim=2) +
                                                 smooth)))
    
    return loss.mean()


def jeccard_sim(pred, target, jec_p=.5):
    """
    Computes the Jaccard Similarity of the semantic segmentation

    Arguments:
        pred: predicted segmentation output by network
        target: binary segmentation targets
        jec_p: prediction cutoff for foreground vs background
    """
    with torch.no_grad():
        pred = pred > jec_p
        intersection = (pred[target > jec_p]).float().sum()
        union = pred.float().sum() + target.float().sum() - intersection
        return (intersection / (union + finfo(torch.float).eps)).item()
    

def mAP(pred, target, cutoff=.5):
    """
    Computes the Average Precision of the semantic segmentation

    Arguments:
        pred: predicted segmentation output by network
        target: binary segmentation targets
        cutoff: prediction cutoff for foreground vs background
    """
    with torch.no_grad():
        pred = pred > cutoff
        intersection = (pred[target > cutoff]).float().sum()
        return (intersection / (pred.float().sum() +
                                finfo(torch.float).eps)).item()
