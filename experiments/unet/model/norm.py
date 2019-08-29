"""
Released under BSD 3-Clause License, 
Modifications are Copyright (c) 2019 Cerebras, Inc.
All rights reserved.
"""
import warnings

import torch
import torch.nn as nn

from online_norm_pytorch import OnlineNorm2D


class Identity(nn.Module):
    __constants__ = []

    def __init__(self, **kwargs):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


def norm(num_features, mode='batch', eps=1e-05, momentum=0.1,
         weight=True, bias=True, track_running_stats=True, **kwargs):
    """
    Function which instantiates a normalization scheme based on mode

    Arguments:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        mode: Option to select normalization method (Default: None)
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        weight: a boolean value that when set to ``True``, this module has
            learnable linear parameters. Default: ``True``
        bias: a boolean value that when set to ``True``, this module has
            learnable bias parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Argument valid when
            using batch norm. Default: ``True``

        Note:
            1. When using BN affine = weight & bias
            2. When using OnlineNorm **kwargs will hold hfwd, hbkw, ctrl_norm
            layer_scaling, and b_size. See definition of OnlineNorm2D for
            specifics.
    """

    if mode == 'batch':
        warnings.warn('Normalizer: Batch')
        affine = weight and bias
        if weight != bias:
            warnings.warn('affine not used in batch norm')
        normalizer = nn.BatchNorm2d(num_features=num_features, eps=eps,
                                    momentum=momentum, affine=affine,
                                    track_running_stats=track_running_stats)

    elif mode == 'online':
        warnings.warn('Normalizer: Online')
        normalizer = OnlineNorm2D(num_features=num_features, eps=eps,
                                  weight=weight, bias=bias, **kwargs)

    elif mode == 'none' or mode is None:
        warnings.warn('Normalizer: None')
        normalizer = Identity()
    
    else:
        raise KeyError('mode options include: "batch" | "online" | "none"')

    return normalizer
