# -*- coding: utf-8 -*-
"""
Released under BSD 3-Clause License,
Copyright (c) 2019 Cerebras Systems Inc.
All rights reserved.

This module implements the Online Normalization algorithm and the components
which go into it.
"""
import warnings

import torch
import torch.nn as nn

from online_norm_pytorch import _C


class LayerScaling(nn.Module):
    r"""Scales inputs by the root of the second moment for groups of channels.
    
    .. math::
        y_g = \frac{x_g}{\sqrt{\mathrm{E}[x_g^2] + \epsilon}}
    
    Args:
        group_size: size of groups
            Default: -1 (no grouping, use all channels)
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> ls = LayerScaling()
        >>> input = torch.randn(64, 128, 32, 32)
        >>> output = ls(input)
    """
    def __init__(self, group_size=-1, eps=1e-5, **kwargs):
        super(LayerScaling, self).__init__()
        self.eps = eps
        self.group_size = group_size

    def extra_repr(self):
        s = (f'eps={self.eps}, group_size={self.group_size}')
        return s

    def forward(self, input):
        shape = input.shape
        self.group_size = shape[1] if self.group_size == -1 else self.group_size
        tmp = input.view(
            shape[0],
            shape[1] // self.group_size,
            self.group_size,
            *shape[2:]
        )
        moment2 = torch.mean(tmp * tmp, dim=[2, 3, 4], keepdim=True)
        out = tmp / torch.sqrt(moment2 + self.eps)
        out = out.view(shape)

        return out


class ActivationClamp(nn.Module):
    r"""Clips the output of CN.

    .. math::
        y = clip(x, -clamp_value, clamp_value)

    Args:
        clamp_value: the value to which activations are clipped.
            Default: 5

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> ac = ActivationClamp(clamp_value)
        >>> input = torch.randn(64, 128, 32, 32)
        >>> output = ac(input)
    """
    def __init__(self, clamp_value=5, **kwargs):
        super(ActivationClamp, self).__init__()
        self.clamp_value = clamp_value

    def extra_repr(self):
        return f'clamp_value={self.clamp_value}'

    def forward(self, input):
        return torch.clamp(input, -self.clamp_value, self.clamp_value)


class Norm2d(nn.Module):
    r"""Applies Normalization (the per-channel exponential moving
    average, ema, forward and control process backward part of the Online
    Normalization algorithm) over a 4D input (a mini-batch of 3D inputs) as
    described in the paper:
    `Online Normalization for Training Neural Networks`.

    .. math::
        y_t = \frac{x_t - \mu_{t-1}}{\sqrt{\sigma^2_{t-1} + \epsilon}}

        \sigma^2_t = (
            \alpha_fwd * \sigma^2_{t-1} +
            \alpha_fwd * (1 - \alpha_fwd) * (x_t - \mu_{t-1}) ^ 2
        )
        \mu_t = \alpha_fwd * \mu_{t-1} + (1 - \alpha_fwd) * x_t

    The mean and standard-deviation are estimated per-channel

    Args:
        num_features: :math:`L` from an expected input of size :math:`(N, L)`
        alpha_fwd: the decay factor to be used in fprop to update statistics.
            Default: 1000
        alpha_bkw: the decay factor to be used in fprop to control the
            gradients propagating through the network. Default: 100
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> norm = Norm2d(128, 0.999, 0.99)
        >>> input = torch.randn(64, 128, 32, 32)
        >>> output = norm(input)
    """

    __constants__ = ['m', 'var', 'u', 'v', 'afwd', 'abkw', 'eps']

    def __init__(self, num_features,
                 alpha_fwd=0.999, alpha_bkw=0.99, eps=1e-05, **kwargs):
        super(Norm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.afwd = alpha_fwd
        self.abkw = alpha_bkw

        # self.m and self.var are the streaming mean and variance respectively
        self.register_buffer('m', torch.zeros([num_features], dtype=torch.float))
        self.register_buffer('var', torch.ones([num_features], dtype=torch.float))

        # self.u and self.v are the control variables respectively
        self.register_buffer('u', torch.zeros([num_features], dtype=torch.float))
        self.register_buffer('v', torch.zeros([num_features], dtype=torch.float))
        self.init_norm_params()

        class Normalization(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                (out, scale, self.m, self.var) = _C.norm_fwd(
                    input, self.m, self.var, self.afwd, self.eps
                )
                ctx.save_for_backward(out, scale,)
                return out

            @staticmethod
            def backward(ctx, grad_out):
                out, scale, = ctx.saved_tensors
                (grad_in, self.u, self.v) = _C.norm_bwd(
                    grad_out, self.u, self.v, out, scale, self.abkw
                )
                return grad_in

        self.normalizer = Normalization.apply

    def init_norm_params(self):
        nn.init.constant_(self.m, 0)
        nn.init.constant_(self.var, 1)
        nn.init.constant_(self.u, 0)
        nn.init.constant_(self.v, 0)

    def moments(self, inputs):
        n = inputs.size(2) * inputs.size(3)

        mu = torch.sum(inputs, dim=(2, 3), keepdim=True) / n
        mu0 = inputs - mu
        return (mu.squeeze(),
                torch.sum(mu0 * mu0, dim=(2, 3), keepdim=False) / n)

    def mean(self, inputs, dim=(1, 2)):
        n = inputs.size(dim[0]) * inputs.size(dim[1])
        return torch.sum(inputs, dim=dim, keepdim=False) / n

    def extra_repr(self):
        s = (f'num_features={self.num_features}, '
             f'afwd={self.afwd}, abkw={self.abkw}, eps={self.eps}')
        return s

    def forward(self, input):
        if self.training:
            return self.normalizer(input)
        mu = self.m.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        std = torch.sqrt(
            self.var.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) + self.eps
        ).type(input.type())
        return (input - mu) / std


class OnlineNorm2d(nn.Module):
    r"""Applies Online Normalization over a 4D input (a mini-batch of 3D
    inputs) as described in the paper:
    `Online Normalization for Training Neural Networks`.

    .. math::
        y_t = LayerScaling(Norm2d(x_t) * \gamma + \beta)

    Args:
        num_features: :math:`L` from an expected input of size :math:`(N, L)`
        alpha_fwd: the decay factor to be used in fprop to update statistics.
            Default: 0.999
        alpha_bkw: the decay factor to be used in fprop to control the gradients
            propagating through the network. Default: 0.99
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters (weight & bias). Default: ``True``
        ecm: a string which defines the error checking mechanism in OnlineNorm.
            Choice: `ac` (Activation Clamping) | `ls` (Layer Scaling).
            Default: ls
        ls_eps: if ecm is `ls`, this is the `ls` eps. Default: 1e-05
        clamp_val: if ecm is `ac` this is the clamp value. Default: 5

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> norm = OnlineNorm2d(128)
        >>> # Without Learnable Parameters
        >>> norm = OnlineNorm2d(128, affine=False)
        >>> input = torch.randn(64, 128, 32, 32)
        >>> output = norm(input)
    """
    __constants__ = ['weight', 'bias']

    def __init__(self, num_features, alpha_fwd=0.999, alpha_bkw=0.99, eps=1e-05,
                 affine=True, ecm='ac', ls_eps=1e-05, clamp_val=5, **kwargs):
        super(OnlineNorm2d, self).__init__()
        self.num_features = num_features

        self.norm = Norm2d(num_features,
                           alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw, eps=eps)

        if ecm.lower() == 'ls':
            self.ecm = LayerScaling(eps=ls_eps, **kwargs)
            warnings.warn('Using LayerScaling in Online Normalization')
        elif ecm.lower() == 'ac':
            self.ecm = ActivationClamp(clamp_val)
            warnings.warn('Using ActivationClamping in Online Normalization')
        else:
            warnings.warn(
                'No guards on statistical estimates of OnlineNorm, '
                'possible options: ls | ac'
            )
            self.ecm = None

        if affine:
            self.weight = nn.Parameter(torch.ones([num_features]),
                                       requires_grad=True)
            self.bias = nn.Parameter(torch.zeros([num_features]),
                                     requires_grad=True)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def extra_repr(self):
        return f'weight={self.weight is not None}, bias={self.bias is not None}'

    def forward(self, input):
        # apply norm
        out = self.norm(input)
        # scale output
        if self.weight is not None:
            out = out * self.weight.type(out.type()).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # add bias
        if self.bias is not None:
            out = out + self.bias.type(out.type()).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # guard for numerical stability of statistical estimates in OnlineNorm
        return self.ecm(out) if self.ecm is not None else out
