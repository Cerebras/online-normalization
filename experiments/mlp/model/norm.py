"""
Released under BSD 3-Clause License,
Copyright (c) 2019 Cerebras, Inc.
All rights reserved.
"""
import warnings

import torch
import torch.nn as nn

from online_norm_pytorch import OnlineNorm1d


class Identity(nn.Module):
    __constants__ = []

    def __init__(self, **kwargs):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


class LayerNorm1d(nn.Module):
    __constants__ = ['weight', 'bias']

    def __init__(self, num_features, eps=1e-05, affine=True, **kwargs):
        super(LayerNorm1d, self).__init__()

        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones([num_features]),
                                       requires_grad=True)
            self.bias = nn.Parameter(torch.zeros([num_features]),
                                     requires_grad=True)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        return nn.functional.layer_norm(x, x.shape[1],
                                        weight=self.weight, bias=self.bias,
                                        eps=self.eps)


def norm(num_features, mode='batch', eps=1e-05, momentum=0.1, affine=True,
         track_running_stats=True,
         batch_size=None, alpha_fwd=0.999, alpha_bkw=0.99,
         ecm='ls', ls_eps=1e-05, clamp_val=5, **kwargs):
    """
    Function which instantiates a normalization scheme based on mode

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        mode: Option to select normalization method (Default: batch)
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters (weight & bias). Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Argument valid when
            using batch norm. Default: ``True``
    
    OnlineNorm Args:
        batch_size: Deprecated with Norm1DBatched. in order to speed up computation
            we need to know and fix the batch size a priori.
        alpha_fwd: the decay factor to be used in fprop to update statistics.
            Default: 0.999
        alpha_bkw: the decay factor to be used in fprop to control the gradients
            propagating through the network. Default: 0.99
        ecm: a string which defines the error checking mechanism in OnlineNorm.
            Choice: `ac` (Activation Clamping) | `ls` (Layer Scaling).
            Default: ls
        ls_eps: if ecm is `ls`, this is the `ls` eps. Default: 1e-05
        clamp_val: if ecm is `ac` this is the clamp value. Default: 5
    """

    if mode == 'batch':
        warnings.warn('Normalizer: Batch')
        normalizer = nn.BatchNorm1d(num_features=num_features, eps=eps,
                                    momentum=momentum, affine=affine,
                                    track_running_stats=track_running_stats)

    elif mode == 'layer':
        warnings.warn('Normalizer: Layer')
        normalizer = LayerNorm1d(num_features, eps=eps, affine=affine)

    elif mode == 'online':
        warnings.warn('Normalizer: Online')
        normalizer = OnlineNorm1d(num_features, batch_size=batch_size,
                                  alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw, 
                                  eps=eps, affine=affine, ecm=ecm,
                                  ls_eps=ls_eps, clamp_val=clamp_val, **kwargs)

    elif mode == 'none' or mode is None:
        warnings.warn('Normalizer: None')
        normalizer = Identity()

    else:
        raise KeyError('mode options include: '
                       '"batch" | "layer" | "online" | "none"')

    return normalizer
