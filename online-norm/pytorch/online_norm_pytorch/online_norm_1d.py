#!/usr/bin/env python
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


class LayerScaling1d(nn.Module):
    r"""Scales inputs by the root of the second moment for groups.
    
    .. math::
        y_g = \frac{x_g}{\sqrt{\mathrm{E}[x_g^2] + \epsilon}}
    
    Args:
        group_size: size of groups
            Default: -1 (no grouping, use all channels)
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5

    Shape:
        - Input: :math:`(N, C)`
        - Output: :math:`(N, C)` (same shape as input)

    Examples::

        >>> ls = LayerScaling1d()
        >>> input = torch.randn(64, 128)
        >>> output = ls(input)
    """
    def __init__(self, group_size=-1, eps=1e-5):
        super(LayerScaling1d, self).__init__()
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
            self.group_size
        )
        moment2 = torch.mean(tmp * tmp, dim=[2], keepdim=True)
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
        - Input: :math:`(N, C)`
        - Output: :math:`(N, C)` (same shape as input)

    Examples::

        >>> ac = ActivationClamp(clamp_value)
        >>> input = torch.randn(64, 128)
        >>> output = ac(input)
    """
    def __init__(self, clamp_value=5):
        super(ActivationClamp, self).__init__()
        self.clamp_value = clamp_value

    def extra_repr(self):
        return f'clamp_value={self.clamp_value}'

    def forward(self, input):
        return torch.clamp(input, -self.clamp_value, self.clamp_value)


class Norm1d(nn.Module):
    r"""Applies Normalization (the per feature exponential moving
    average, ema, forward and control process backward part of the Online
    Normalization algorithm) over a 2D input (a mini-batch of 1d inputs) as
    described in the paper:
    `Online Normalization for Training Neural Networks`.

    .. math::
        y_t = \frac{x_t - \mu_{t-1}}{\sqrt{\sigma^2_{t-1} + \epsilon}}

        \sigma^2_t = (
            \alpha_fwd * \sigma^2_{t-1} +
            \alpha_fwd * (1 - \alpha_fwd) * (x_t - \mu_{t-1}) ^ 2
        )
        \mu_t = \alpha_fwd * \mu_{t-1} + (1 - \alpha_fwd) * x_t

    The mean and standard-deviation are estimated per-feature

    Args:
        num_features: :math:`L` from an expected input of size :math:`(N, L)`
        alpha_fwd: the decay factor to be used in fprop to update statistics.
            Default: 0.999
        alpha_bkw: the decay factor to be used in fprop to control the gradients
            propagating through the network.
            Default: 0.99
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5

    Shape:
        - Input: :math:`(N, L)`
        - Output: :math:`(N, L)` (same shape as input)

    Examples::

        >>> norm = Norm1d(100, 0.999, 0.99)
        >>> input = torch.randn(20, 100)
        >>> output = norm(input)
    """

    __constants__ = ['m', 'var', 'u', 'v', 'afwd', 'abkw', 'eps']

    def __init__(self, num_features,
                 alpha_fwd=0.999, alpha_bkw=0.99, eps=1e-05):
        super(Norm1d, self).__init__()
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
                    input.contiguous(), self.m, self.var, self.afwd, self.eps
                )
                ctx.save_for_backward(out, scale,)
                return out

            @staticmethod
            def backward(ctx, grad_out):
                out, scale, = ctx.saved_tensors
                (grad_in, self.u, self.v) = _C.norm_bwd(
                    grad_out.contiguous(), self.u, self.v, out, scale, self.abkw
                )
                return grad_in

        self.normalizer = Normalization.apply

    def init_norm_params(self):
        nn.init.constant_(self.m, 0)
        nn.init.constant_(self.var, 1)
        nn.init.constant_(self.u, 0)
        nn.init.constant_(self.v, 0)

    def extra_repr(self):
        s = (f'num_features={self.num_features}, '
             f'afwd={self.afwd}, abkw={self.abkw}, eps={self.eps}')
        return s

    def forward(self, input):
        if self.training:
            return self.normalizer(input)
        mu = self.m.type(input.type()).unsqueeze(0)
        std = torch.sqrt(self.var.unsqueeze(0) + self.eps).type(input.type())
        return (input - mu) / std


def lin_momentum(mu_prev, mu_curr, mu_stream,
                 momentum, momentum_pow, momentum_batch):
    """
    Helper function for performing an exponential moving average, ema, using
    convolutions which can be distributed across compute fabric.

    Arguments:
        mu_prev: previous time steps statistic
        mu_curr: this time steps statistic
        mu_stream: ema from last time step
        momentum: 1 - 1 / h
        momentum_pow: momentum ^ range(b_size - 1, -1, -1)
        momentum_batch: momentum ^ b_size of size :math:`(N, L)`

    Return:
        updated mu_stream (stale): to use for fprop
        updated mu_stream (current): to cache for next iteration
    """
    input = torch.cat([mu_prev[1:], mu_curr]).transpose(0, 1).unsqueeze(1)
    tmp = torch.nn.functional.conv1d(input, momentum_pow).squeeze().transpose(0, 1)
    curr = (momentum_batch * mu_stream + (1 - momentum) * tmp)

    return torch.cat([mu_stream[-1].unsqueeze(0), curr[:-1]]), curr


def conv_alongb_w1(input, b, c):
    """
    helper functions for fast lin_crtls
    Convolve along 2b dimension with a b length vector of 1's

    assumes order (b, 2b, c)
    """
    input = input.transpose(0, 1).clone().view(2 * b, -1).t().unsqueeze(1)
    tmp = torch.nn.functional.conv1d(input, torch.ones_like(input[0:1, 0:1, :b]))
    return tmp.squeeze().t().view(b + 1, b, c).clone().transpose(0, 1)


def lin_crtl(delta, out, b_size, num_features, v_p, alpha_p, beta_p,
             abkw, eps=1e-32):
    """
    Helper function for controlling with v controller using
    convolutions which can be distributed across compute fabric.

    Arguments:
        delta: grad_out
        out: output of normalization
        b_size: batch size
        num_features: number of features (L)
        v_p: the previous time steps v estimate
        alpha_p: the previous time steps alpha estimate
        beta_p: the previous time steps beta estimate
        abkw: decay factor for bpass
        eps: eps by which to clip alpha which gets log applied to it.
            Default: 1e-32

    Return:
        grad_in
        v_new: current estimate of v
        alpha_p: current estimate of alpha
        beta_p: current estimate of beta
    """

    # expect 0 << alpha ~<1 so we can move it to log space
    alpha = torch.ones_like(delta[:, :]) - (1 - abkw) * out * out
    alpha = torch.clamp(alpha, min=eps)
    beta = delta * out

    alpha2log = torch.log(torch.cat((alpha_p, alpha), 0))
    beta2 = torch.cat((beta_p, beta), 0)

    alpha2logcir = alpha2log.repeat(2 * b_size + 1, 1).view(
        2 * b_size, 2 * b_size + 1, num_features)[1:b_size + 1, :b_size]

    alpha2logcir2 = torch.cat((alpha2logcir,
                               torch.zeros_like(alpha2logcir)), 1)
    alpha2logcir2conv = conv_alongb_w1(alpha2logcir2, b_size, num_features)
    weight_d = torch.exp(alpha2logcir2conv)

    beta2cir = beta2.repeat(2 * b_size + 1, 1).view(
        2 * b_size, 2 * b_size + 1, num_features)[1:b_size + 1, :b_size]

    v_prev = torch.cat((v_p.reshape(b_size, 1, num_features), beta2cir), 1)

    v_new = (weight_d * v_prev).sum(1)

    alpha_p = alpha
    beta_p = beta

    vp = torch.cat((v_p[-1].unsqueeze(0), v_new[:-1]), 0)

    return (
        delta - vp.view(b_size, num_features) * (1 - abkw) * out,
        v_new,
        alpha_p,
        beta_p
    )


class Norm1dBatched(nn.Module):
    r"""Applies Normalization (the per feature exponential moving
    average, ema, forward and control process backward part of the Online
    Normalization algorithm) over a 2D input (a mini-batch of 1D inputs) as
    described in the paper:
    `Online Normalization for Training Neural Networks`. This class implements
    a version of the mathematics below which uses linear operations and can
    therefore be distributed on a GPU.

    .. math::
        y_t = \frac{x_t - \mu_{t-1}}{\sqrt{\sigma^2_{t-1} + \epsilon}}

        \sigma^2_t = (
            \alpha_fwd * \sigma^2_{t-1} +
            \alpha_fwd * (1 - \alpha_fwd) * (x_t - \mu_{t-1}) ^ 2
        )

        \mu_t = \alpha_fwd * \mu_{t-1} + (1 - \alpha_fwd) * x_t

    The mean and standard-deviation are estimated per-feature.

    The math above represents the calculations occurring in the layer. To
    speed up computation with batched training we linearize the computation
    along the batch dimension and use convolutions in place of sums to
    distribute the computation across compute fabric.

    Args:
        num_features: :math:`L` from an expected input of size :math:`(N, L)`
        batch_size (N): in order to speed up computation we need to know and fix
            the batch size a priori.
        alpha_fwd: the decay factor to be used in fprop to update statistics.
            Default: 0.999
        alpha_bkw: the decay factor to be used in fprop to control the gradients
            propagating through the network.
            Default: 0.99
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5

    Shape:
        - Input: :math:`(N, L)`
        - Output: :math:`(N, L)` (same shape as input)

    Examples::
        >>> B, C = 32, 256
        >>> norm = Norm1dBatched(C, B, 0.999, 0.99)
        >>> input = torch.randn(B, C)
        >>> output = norm(input)
    """
    __constants__ = ['m', 'var', 'u', 'v', 'm_p', 'var_p', 'u_p', 'v_p',
                     'beta_p', 'alpha_p', 'afwd', 'abkw', 'eps']

    def __init__(self, num_features, batch_size,
                 alpha_fwd=0.999, alpha_bkw=0.99, eps=1e-05):
        super(Norm1dBatched, self).__init__()
        
        warnings.warn(
                'Deprecated Warning: Use kernel instead.'
            )

        assert isinstance(batch_size, int), 'batch_size must be an integer'
        assert batch_size > 0, 'batch_size must be greater than 0'
        self.num_features = num_features
        self.batch_size = batch_size
        self.eps = eps

        self.afwd = alpha_fwd
        self.abkw = alpha_bkw

        # batch streaming parameters fpass
        self.af_pow = None
        self.af_batch = None

        # batch streaming parameters bpass
        self.ab_pow = None
        self.ab_batch = None

        # self.m and self.var are the streaming mean and variance respectively
        self.register_buffer('m', torch.zeros([batch_size, num_features]))
        self.register_buffer('var', torch.ones([batch_size, num_features]))
        self.register_buffer('m_p', torch.zeros([batch_size, num_features]))
        self.register_buffer('var_p', torch.ones([batch_size, num_features]))

        # self.u and self.v are the control variables respectively
        self.register_buffer('u', torch.zeros([batch_size, num_features]))
        self.register_buffer('u_p', torch.zeros([batch_size, num_features]))
        self.register_buffer('v_p', torch.zeros([batch_size, num_features]))
        self.register_buffer('beta_p', torch.zeros([batch_size, num_features]))
        self.register_buffer('alpha_p', torch.ones([batch_size, num_features]))
        self.init_norm_params()

        class Normalization(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):

                if self.af_pow is None:
                    range_b = torch.arange(self.batch_size - 1, -1, -1).type(input.type())
                    self.af_pow = (self.afwd ** range_b).view(1, 1, -1)
                    self.af_batch = input.new_full((self.batch_size, num_features),
                                                   self.afwd ** self.batch_size)
                    self.ab_pow = (self.abkw ** range_b).view(1, 1, -1)
                    self.ab_batch = input.new_full((self.batch_size, num_features),
                                                   self.abkw ** self.batch_size)

                momentum = self.afwd
                momentum_pow = self.af_pow
                momentum_batch = self.af_batch

                _mu_b, mu_b = lin_momentum(self.m_p, input, self.m, momentum,
                                           momentum_pow, momentum_batch)

                var_current = momentum * (input - _mu_b) ** 2
                _var_b, var_b = lin_momentum(self.var_p, var_current, self.var,
                                             momentum, momentum_pow,
                                             momentum_batch)

                scale = torch.sqrt(_var_b + self.eps)
                out = (input - _mu_b) / scale
                ctx.save_for_backward(out, scale)

                self.m_p.data = input.clone()
                self.var_p.data = var_current.clone()

                self.m.data = mu_b.clone()
                self.var.data = var_b.clone()

                return out

            @staticmethod
            def backward(ctx, grad_in):
                out, scale, = ctx.saved_tensors

                # v controller
                lin_ctrl_out = lin_crtl(grad_in, out,
                                        self.batch_size, self.num_features,
                                        self.v_p, self.alpha_p, self.beta_p,
                                        abkw=self.abkw, eps=1e-5)

                (grad_delta, self.v_p.data,
                 self.alpha_p.data, self.beta_p.data) = lin_ctrl_out

                grad_delta = grad_delta / scale

                # mean (u) controller
                u_tmp = grad_delta
                _u_b, u_b = lin_momentum(self.u_p, u_tmp, self.u, self.abkw,
                                         self.ab_pow, self.ab_batch)
                grad_delta = grad_delta - _u_b
                self.u_p.data, self.u.data = u_tmp.clone(), u_b.clone()

                return grad_delta

        self.normalizer = Normalization.apply

    def init_norm_params(self):
        nn.init.constant_(self.m, 0)
        nn.init.constant_(self.var, 1)
        nn.init.constant_(self.u, 0)
        nn.init.constant_(self.m_p, 0)
        nn.init.constant_(self.var_p, 1)
        nn.init.constant_(self.u_p, 0)
        nn.init.constant_(self.v_p, 0)

        nn.init.constant_(self.beta_p, 0)
        nn.init.constant_(self.alpha_p, 1)

    def extra_repr(self):
        s = (f'num_features={self.num_features}, batch_size={self.batch_size}, '
             f'afwd={self.afwd}, abkw={self.abkw}, eps={self.eps}')
        return s

    def forward(self, input):
        if self.training:
            return self.normalizer(input)
        mu = self.m[-1].unsqueeze(0)
        var = self.var[-1].unsqueeze(0)
        return (input - mu) / torch.sqrt(var + self.eps)


class OnlineNorm1d(nn.Module):
    r"""Applies Online Normalization over a 2D input (a mini-batch of 1d
    inputs) as described in the paper:
    `Online Normalization for Training Neural Networks`.

    .. math::
        y_t = LayerScaling1d(Norm1d(x_t) * \gamma + \beta)

    Args:
        num_features: :math:`L` from an expected input of size :math:`(N, L)`
        batch_size: Deprecated with Norm1DBatched. order to speed up computation
            we need to know and fix the batch size a priori.
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
        - Input: :math:`(N, L)`
        - Output: :math:`(N, L)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> norm = OnlineNorm1d(128)
        >>> # Without Learnable Parameters
        >>> norm = OnlineNorm1d(128, affine=False)
        >>> input = torch.randn(64, 128)
        >>> output = norm(input)
    """
    __constants__ = ['weight', 'bias']

    def __init__(self, num_features, batch_size=None,
                 alpha_fwd=0.999, alpha_bkw=0.99, eps=1e-05,
                 affine=True, ecm='ac', ls_eps=1e-05, clamp_val=5, **kwargs):
        super(OnlineNorm1d, self).__init__()
        self.num_features = num_features

        batch_acceleration = True
        if batch_size is None or batch_size == 1:
            batch_acceleration = False

        if batch_acceleration:
            self.norm = Norm1dBatched(num_features, batch_size,
                                      alpha_fwd=alpha_fwd,
                                      alpha_bkw=alpha_bkw,
                                      eps=eps)
        else:
            self.norm = Norm1d(num_features,
                               alpha_fwd=alpha_fwd,
                               alpha_bkw=alpha_bkw,
                               eps=eps)

        if ecm.lower() == 'ls':
            self.ecm = LayerScaling1d(eps=ls_eps, **kwargs)
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
            out = out * self.weight.type(out.type()).unsqueeze(0)
        # add bias
        if self.bias is not None:
            out = out + self.bias.type(out.type()).unsqueeze(0)
        # guard for numerical stability of statistical estimates in OnlineNorm
        return self.ecm(out) if self.ecm is not None else out
