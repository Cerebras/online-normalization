# -*- coding: utf-8 -*-
"""
Released under BSD 3-Clause License,
Copyright (c) 2019 Cerebras Systems Inc.
All rights reserved.

This is a numpy implementation of the Online Normalization algorithm and the 
components which go into it.
"""
import warnings
import numpy as np


def norm_forward(inputs, mstream, varstream, afwd, eps):
    """
    Implements the forward pass of the norm op
    
    For each incoming sample it does:
        out = (inputs - mstream) / sqrt(varstream)
        varstream = afwd * varstream + 
                        (afwd * (1 - afwd) * (inputs - mstream) ** 2
        mstream = afwd * mstream + (1 - afwd) * inputs
    """
    center = np.empty_like(inputs)
    scale = np.empty_like(inputs)

    for idx in range(inputs.shape[0]):
        # fprop activations
        center[idx] = mstream.copy()
        scale[idx] = np.sqrt(varstream + eps).copy()

        # Update statistics trackers
        varstream = (afwd * varstream + 
                     (afwd * (1 - afwd) * (inputs[idx] - mstream) ** 2))
        mstream += ((1 - afwd) * (inputs[idx] - mstream))

    out = (inputs - center) / scale
    cache = (out, scale)

    return out, mstream, varstream, cache


def norm_backward(grad_out, ustream, vstream, abkw, cache):
    """
    Implements the backwards pass of the norm op
    
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
        grad = grad_out[idx] - (1 - abkw) * vstream * out[idx]
        vstream += grad * out[idx]
        grad = grad / scale[idx]
        grad_in[idx] = grad - (1 - abkw) * ustream
        ustream += grad_in[idx]

    return grad_in, ustream, vstream, (None, )


def mult_scale_forward(inputs, weight):
    cache = (inputs, weight)
    return inputs * weight[np.newaxis, :], cache


def mult_scale_backward(grad_out, cache):
    inputs, weight = cache
    grad_in = grad_out * weight[np.newaxis, :]
    grad_weight = (grad_out * inputs).sum(0)
    return grad_in, (grad_weight, )


def add_bias_forward(inputs, bias):
    return inputs + bias[np.newaxis, :], None


def add_bias_backward(grad_out, cache):
    grad_in = grad_out
    grad_bias = grad_out.sum(0)
    return grad_in, (grad_bias, )


def layer_scaling_forward(inputs, eps):
    moment2 = np.mean(inputs * inputs, axis=1, keepdims=True)
    scale = np.sqrt(moment2 + eps)
    out = inputs / scale
    cache = (out, scale)
    return out, cache


def layer_scaling_backward(grad_out, cache):
    out, scale = cache
    proj = np.mean(grad_out * out, axis=1, keepdims=True) * out
    grad_in = (grad_out - proj) / scale
    return grad_in, (None, )


def activation_clamping_forward(inputs, clamp_val):
    out = np.clip(inputs, -clamp_val, clamp_val)
    cache = (out, clamp_val)
    return out, cache


def activation_clamping_backward(grad_out, cache):
    out, clamp_val = cache
    grad_in = grad_out.copy()
    grad_in[np.where(out == clamp_val)] = 0
    grad_in[np.where(out == -clamp_val)] = 0
    return grad_in, (None, )


class Norm1d:
    r"""Applies Normalization (the per-channel exponential moving
    average, ema, forward and control process backward part of the Online
    Normalization algorithm) over a 2D inputs (a mini-batch of 1D inputs) as
    described in the paper:
    `Online Normalization for Training Neural Networks`.

    .. math::
        y_t = \frac{x_t - \mu_{t-1}}{\sqrt{\sigma^2_{t-1} + \epsilon}}

        \sigma^2_t = \alpha_fwd * \sigma^2_{t-1} + \alpha_fwd * (1 - \alpha_fwd) * (x_t - \mu_{t-1}) ^ 2
        \mu_t = \alpha_fwd * \mu_{t-1} + (1 - \alpha_fwd) * x_t

    The mean and standard-deviation are estimated per-channel

    Args:
        num_features: :math:`L` from an expected inputs of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        alpha_fwd: the decay factor to be used in fprop to update statistics.
            Default: 0.999
        alpha_bkw: the decay factor to be used in fprop to control the gradients
            propagating through the network. Default: 0.99

    Shape:
        - Input: :math:`(N, L)`
        - Output: :math:`(N, L)` (same shape as inputs)

    Examples::

        >>> norm = Norm1d(128, .999, .99)
        >>> inputs = numpy.random.randn(64, 128)
        >>> output = norm(inputs)
    """

    __constants__ = ['m', 'var', 'u', 'v', 'afwd', 'abkw', 'eps']

    def __init__(self, num_features,
                 alpha_fwd=0.999, alpha_bkw=0.99, eps=1e-05, **kwargs):
        super(Norm1d, self).__init__()
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
            out, self.m, self.var, self.cache = norm_forward(inputs,
                                                             self.m,
                                                             self.var,
                                                             self.afwd,
                                                             self.eps)
            return out
        mu = self.m[np.newaxis, :]
        var = self.var[np.newaxis, :]
        return (inputs - mu) / np.sqrt(var + self.eps)

    def backward(self, grad_out):
        grad_in, self.u, self.v, _ = norm_backward(grad_out,
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
        num_features: :math:`C` from an expected inputs of size :math:`(N, C)`

    Shape:
        - Input: :math:`(N, C)`
        - Output: :math:`(N, C)` (same shape as inputs)

    Examples::

        >>> ws = WeightScale(128)
        >>> inputs = numpy.random.randn(64, 128)
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

        >>> ab = AddBias(C)
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
        >>> inputs = numpy.random.randn(64, 128)
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
        grad_in, _ = layer_scaling_backward(grad_out, self.cache)
        return grad_in


class ActivationClamp:
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
        >>> input = numpy.random.randn(64, 128)
        >>> output = ac(input)
    """
    def __init__(self, clamp_value=5, **kwargs):
        super(ActivationClamp, self).__init__()
        self.clamp_val = clamp_value
        self.opt_params = None

    def __call__(self, inputs):
        output, self.cache = activation_clamping_forward(inputs, self.clamp_val)
        return output

    def backward(self, grad_out):
        grad_in, _ = activation_clamping_backward(grad_out, self.cache)
        return grad_in


class OnlineNorm1d:
    r"""Applies Online Normalization over a 2D inputs (a mini-batch of 1D
    inputs) as described in the paper:
    `Online Normalization for Training Neural Networks`.

    .. math::
        y_t = LayerScaling(Norm1d(x_t) * \gamma + \beta)

    Args:
        num_features: :math:`L` from an expected inputs of size :math:`(N, L)`
        alpha_fwd: the decay factor to be used in fprop to update statistics.
            Default: 0.999
        alpha_bkw: the decay factor to be used in fprop to control the gradients
            propagating through the network. Default: 0.99
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters (weight & bias). Default: ``True``
        ecm: a string which defines the error compensation mechanism in OnlineNorm.
            Choice: `ac` (Activation Clamping) | `ls` (Layer Scaling).
        ls_eps: if ecm is `ls`, this is the `ls` eps.
        clamp_val: if ecm is `ac` this is the clamp value.

    Shape:
        - Input: :math:`(N, L)`
        - Output: :math:`(N, L)` (same shape as inputs)

    Examples::

        >>> # With Learnable Parameters
        >>> norm = OnlineNorm1d(128, .999, .99)
        >>> # Without Learnable Parameters
        >>> norm = OnlineNorm1d(128, .999, .99, affine=False)
        >>> inputs = numpy.random.randn(64, 128)
        >>> output = norm(inputs)
    """
    __constants__ = ['weight', 'bias']

    def __init__(self, num_features, alpha_fwd=0.999, alpha_bkw=0.99,
                 eps=1e-05, affine=True,
                 ecm='ac', ls_eps=1e-05, clamp_val=5, **kwargs):
        super(OnlineNorm1d, self).__init__()
        self.num_features = num_features

        self.norm = Norm1d(num_features,
                           alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw,
                           eps=eps, **kwargs)

        if affine:
            self.weight = WeightScale(num_features)
            self.bias = AddBias(num_features)
        else:
            self.weight = None
            self.bias = None

        if ecm.lower() == 'ls':
            self.ecm = LayerScaling(eps=ls_eps)
            warnings.warn('Using LayerScaling in Online Normalization')
        elif ecm.lower() == 'ac':
            self.ecm = ActivationClamp(clamp_val=clamp_val)
            warnings.warn('Using ActivationClamping in Online Normalization')
        else:
            warnings.warn(
                'No guards on statistical estimates of OnlineNorm,'
                'possible options: ls | ac'
            )
            self.ecm = None

        self.opt_params = None

    def __call__(self, inputs):
        # apply norm
        out = self.norm(inputs)
        
        # affine transform (optional)
        out = self.weight(out) if self.weight is not None else out  # scale output
        out = self.bias(out) if self.bias is not None else out  # add bias

        # apply ecm (optional)
        return self.ecm(out) if self.ecm is not None else out

    def backward(self, grad_out):
        grad = grad_out
        grad = self.ecm.backward(grad) if self.ecm is not None else grad

        grad = self.bias.backward(grad) if self.bias is not None else grad
        grad = self.weight.backward(grad) if self.weight is not None else grad

        grad_in = self.norm.backward(grad)

        return grad_in
