# -*- coding: utf-8 -*-
"""
Released under BSD 3-Clause License, 
Copyright (c) 2019 Cerebras Systems Inc.
All rights reserved.

This is a numpy implementation of the Online Normalization algorithm and the 
components which go into it.
"""
import numpy as np


def control_norm_forward(inputs, mstream, varstream, afwd, eps):
    """
    Implements the forward pass of the control norm
    
    For each incoming sample it does:
        out = (inputs - mstream) / sqrt(varstream)
        varstream = afwd * varstream + (1 - afwd) * var(x) +
                        (afwd * (1 - afwd) * (mu(x) - mstream) ** 2
        mstream = afwd * mstream + (1 - afwd) * mu(x)
    """
    center = np.empty_like(inputs[:, :, 0, 0])
    scale = np.empty_like(inputs[:, :, 0, 0])

    mu = np.mean(inputs, axis=(2, 3), keepdims=False)
    centered = inputs - mu[:, :, np.newaxis, np.newaxis]
    var = np.mean(centered * centered, axis=(2, 3), keepdims=False)

    for idx in range(inputs.shape[0]):
        # fprop activations
        center[idx] = mstream.copy()
        scale[idx] = np.sqrt(varstream + eps).copy()

        # Update statistics trackers
        varstream = (afwd * varstream + (1 - afwd) * var[idx] +
                     (afwd * (1 - afwd) * (mu[idx] - mstream) ** 2))
        mstream += ((1 - afwd) * (mu[idx] - mstream))

    out = ((inputs - center[:, :, np.newaxis, np.newaxis]) / 
           scale[:, :, np.newaxis, np.newaxis])
    cache = (out, scale)

    return out, mstream, varstream, cache


def control_norm_backward(grad_out, ustream, vstream, abkw, cache):
    """
    Implements the forward pass of the control norm
    
    For each incoming sample it does:
        grad = grad_out - (1 - abkw) * vstream * out
        vstream = vstream + mu()

        y = (x - mstream) / sqrt(varstream)
        varstream = afwd * varstream + (1 - afwd) * var(x) +
                        (afwd * (1 - afwd) * (mu(x) - mstream) ** 2
        mstream = afwd * mstream + (1 - afwd) * mu(x)
    """
    out, scale = cache

    grad_in = np.empty_like(grad_out)

    for idx in range(grad_out.shape[0]):
        grad = (grad_out[idx] -
                (1 - abkw) * vstream[:, np.newaxis, np.newaxis] * out[idx])
        vstream += np.mean(grad * out[idx], axis=(1, 2), keepdims=False)
        grad = grad / scale[idx, :, np.newaxis, np.newaxis]
        grad_in[idx] = grad - (1 - abkw) * ustream[:, np.newaxis, np.newaxis]
        ustream += np.mean(grad_in[idx], axis=(1, 2), keepdims=False)

    return grad_in, ustream, vstream, (None, )


def mult_scale_forward(inputs, weight):
    cache = (inputs, weight)
    return inputs * weight[np.newaxis, :, np.newaxis, np.newaxis], cache


def mult_scale_backward(grad_out, cache):
    inputs, weight = cache
    grad_in = grad_out * weight[np.newaxis, :, np.newaxis, np.newaxis]
    grad_weight = (grad_out * inputs).sum((0, 2, 3))
    return grad_in, (grad_weight, )


def add_bias_forward(inputs, bias):
    return inputs + bias[np.newaxis, :, np.newaxis, np.newaxis], None


def add_bias_backward(grad_out, cache):
    grad_in = grad_out
    grad_bias = grad_out.sum((0, 2, 3))
    return grad_in, (grad_bias, )


def layer_scaling_forward(inputs, eps):
    axis = tuple(range(inputs.ndim))
    moment2 = np.mean(inputs * inputs, axis=axis[1:], keepdims=True)
    scale = np.sqrt(moment2 + eps)
    out = inputs / scale
    cache = (out, scale, axis)
    return out, cache


def layer_scaling_backward(grad_out, cache):
    out, scale, axis = cache
    proj = np.mean(grad_out * out, axis=axis[1:], keepdims=True) * out
    grad_in = (grad_out - proj) / scale
    return grad_in, (None, )


class ControlNorm2d:
    r"""Applies Control Normalization (the per-channel exponential moving
    average, ema, forward and control process backward part of the Online
    Normalization algorithm) over a 4D inputs (a mini-batch of 3D inputs) as
    described in the paper:
    `Online Normalization for Training Neural Networks`.

    .. math::
        y_t = \frac{x_t - \mu_{t-1}}{\sqrt{\sigma^2_{t-1} + \epsilon}}

        \sigma^2_t = \alpha_fwd * \sigma^2_{t-1} + (1 - \alpha_fwd) * var(x_t) + \alpha_fwd * (1 - \alpha_fwd) * (x_t - \mu_{t-1}) ^ 2
        \mu_t = \alpha_fwd * \mu_{t-1} + (1 - \alpha_fwd) * mu(x_t)

    The mean and standard-deviation are estimated per-channel

    Args:
        num_features: :math:`C` from an expected inputs of size :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        alpha_fwd: the decay factor to be used in fprop to update statistics.
            Default: 0.999
        alpha_bkw: the decay factor to be used in fprop to control the gradients
            propagating through the network. Default: 0.99

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as inputs)

    Examples::

        >>> norm = ControlNorm2d(128, .999, .99)
        >>> inputs = numpy.random.randn(64, 128, 32, 32)
        >>> output = norm(inputs)
    """

    __constants__ = ['m', 'var', 'u', 'v', 'afwd', 'abkw', 'eps']

    def __init__(self, num_features,
                 alpha_fwd=.999, alpha_bkw=.99, eps=1e-05, **kwargs):
        super(ControlNorm2d, self).__init__()
        self.training = True
        self.num_features = num_features
        self.eps = eps
        self.opt_params = None

        self.afwd = alpha_fwd
        self.abkw = alpha_bkw

        self.m = np.zeros(num_features)
        self.var = np.ones(num_features)
        self.u = np.zeros(num_features)
        self.v = np.zeros(num_features)

    def __call__(self, inputs):
        if self.training:
            out, self.m, self.var, self.cache = control_norm_forward(inputs,
                                                                     self.m,
                                                                     self.var,
                                                                     self.afwd,
                                                                     self.eps)
        mu = self.m[np.newaxis, :, np.newaxis, np.newaxis]
        var = self.var[np.newaxis, :, np.newaxis, np.newaxis]
        return (inputs - mu) / np.sqrt(var + self.eps)

    def backward(self, grad_out):
        grad_in, self.u, self.v, grad_param = control_norm_backward(grad_out,
                                                                    self.u,
                                                                    self.v,
                                                                    self.abkw,
                                                                    self.cache)
        return grad_in


class WeightScale:
    r"""Scales inputs by weight parameter
    .. math::

        y = w * x

    Args:
        num_features: :math:`C` from an expected inputs of size :math:`(N, C, H, W)`

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as inputs)

    Examples::

        >>> ws = WeightScale(128)
        >>> inputs = numpy.random.randn(64, 128, 32, 32)
        >>> output = ws(inputs)

    """
    def __init__(self, num_features, **kwargs):
        super(WeightScale, self).__init__()
        self.num_features = num_features

        self.weight = np.ones(num_features)

    def __call__(self, inputs):
        output, self.cache = mult_scale_forward(inputs, self.weight)
        return output

    def backward(self, grad_out):
        grad_in, grad_param = mult_scale_backward(grad_out, self.cache)
        self.opt_params = (self.weight, *grad_param)
        return grad_in


class AddBias:
    r"""Scales inputs by weight parameter
    .. math::

        y = x + b

    Args:
        num_features: :math:`C` from an expected inputs of size :math:`(N, C, H, W)`

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as inputs)

    Examples::

        >>> ab = AddBias(128)
        >>> inputs = numpy.random.randn(64, 128, 32, 32)
        >>> output = ab(inputs)

    """
    def __init__(self, num_features, **kwargs):
        super(AddBias, self).__init__()
        self.num_features = num_features

        self.bias = np.zeros(num_features)

    def __call__(self, inputs):
        output, self.cache = add_bias_forward(inputs, self.bias)
        return output

    def backward(self, grad_out):
        grad_in, grad_param = add_bias_backward(grad_out, self.cache)
        self.opt_params = (self.bias, *grad_param)
        return grad_in


class LayerScaling:
    r"""Scales inputs by the second moment for the entire layer.
    .. math::

        y = \frac{x}{\sqrt{\mathrm{E}[x^2] + \epsilon}}

    Args:
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as inputs)

    Examples::

        >>> ls = LayerScaling()
        >>> inputs = numpy.random.randn(64, 128, 32, 32)
        >>> output = ls(inputs)

    """
    def __init__(self, eps=1e-5, **kwargs):
        super(LayerScaling, self).__init__()
        self.eps = eps
        self.opt_params = None

    def __call__(self, inputs):
        output, self.cache = layer_scaling_forward(inputs, self.eps)
        return output

    def backward(self, grad_out):
        grad_in, grad_param = layer_scaling_backward(grad_out, self.cache)
        return grad_in



class OnlineNorm2d:
    r"""Applies Online Normalization over a 4D inputs (a mini-batch of 3D
    inputs) as described in the paper:
    `Online Normalization for Training Neural Networks`.

    .. math::
        y_t = LayerScaling(ControlNorm2d(x_t) * \gamma + \beta)

    Args:
        num_features: :math:`C` from an expected inputs of size :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        alpha_fwd: the decay factor to be used in fprop to update statistics.
            Default: .999
        alpha_bkw: the decay factor to be used in fprop to control the gradients
            propagating through the network. Default: .99
        b_size: in order to speed up computation we need to know and fix the
            batch size a priori.
        weight: a boolean value that when set to ``True``, this module has
            learnable linear parameters. Default: ``True``
        bias: a boolean value that when set to ``True``, this module has
            learnable bias parameters. Default: ``True``
        layer_scaling: a boolean value that when set to ``True``, this module
            has layer scaling at the end. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as inputs)

    Examples::

        >>> # With Learnable Parameters
        >>> norm = OnlineNorm2d(128, .999, .99)
        >>> # Without Learnable Parameters
        >>> norm = OnlineNorm2d(128, .999, .99, weight=False, bias=False)
        >>> inputs = numpy.random.randn(64, 128, 32, 32)
        >>> output = norm(inputs)
    """
    __constants__ = ['weight', 'bias']

    def __init__(self, num_features, alpha_fwd=.999, alpha_bkw=.99,
                 eps=1e-05, weight=True, bias=True,
                 layer_scaling=True, **kwargs):
        super(OnlineNorm2d, self).__init__()
        self.num_features = num_features

        self.ctrl_norm = ControlNorm2d(num_features,
                                       alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw,
                                       eps=eps, **kwargs)

        self.layer_scaling = LayerScaling(eps=eps) if layer_scaling else None

        if weight:
            self.weight = WeightScale(num_features)
        else:
            self.weight = None
        if bias:
            self.bias = AddBias(num_features)
        else:
            self.bias = None

        self.opt_params = None

    def __call__(self, inputs):
        # apply control norm
        out = self.ctrl_norm(inputs)
        
        # affine transform (optional)
        out = self.weight(out) if self.weight else out  # scale output
        out = self.bias(out) if self.bias else out  # add bias

        # apply layer scaling (optional)
        return self.layer_scaling(out) if self.layer_scaling else out

    def backward(self, grad_out):
        grad = grad_out
        grad = self.layer_scaling.backward(grad) if self.layer_scaling else grad

        grad = self.bias.backward(grad) if self.bias else grad
        grad = self.weight.backward(grad) if self.weight else grad

        grad_in = self.ctrl_norm.backward(grad)

        return grad_in
