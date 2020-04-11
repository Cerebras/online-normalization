"""
Released under BSD 3-Clause License,
Copyright (c) 2019 Cerebras Systems Inc.
All rights reserved.

TensorFlow Implementation of the Online Normalization Layer
"""
import warnings

import tensorflow as tf
from tensorflow.python.framework import dtypes, tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints, initializers, regularizers
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.layers import Layer

try:
    online_norm_kernel = tf.load_op_library('./online_norm_tf/csrc/online_norm_gpu.so')
except:
    online_norm_kernel = tf.load_op_library('./online_norm_tf/csrc/online_norm_cpu.so')

online_norm_fwd = online_norm_kernel.online_norm_fwd
online_norm_u_ctrl = online_norm_kernel.online_norm_u_ctrl
online_norm_v_ctrl = online_norm_kernel.online_norm_v_ctrl
from tensorflow.keras.mixed_precision.experimental import Policy


class Norm(Layer):
    """
    Custom backprop normalizer implementation of the
    [Online Normalization Algorithm](https://arxiv.org/abs/1905.05894)

    Note:
        Implemented with custom gradients, using the @tf.custom_gradient
        decorator which requires tf.__version__ >= 1.7

    Arguments:
        alpha_fwd: the decay factor to be used in fprop to update statistics.
            Default: 0.999
        alpha_bkw: the decay factor to be used in bprop to control the
            gradients propagating through the network. Default: 0.99
        axis: Integer, the axis that should be normalized. Defualt: 1
            Note kernel only supports 1
        epsilon: a value added to the denominator for numerical stability.
            Default: 1e-5.
        stream_mu_initializer: Initializer for the streaming mean.
        stream_var_initializer: Initializer for the streaming variance.
        u_ctrl_initializer: Initializer for the u control variable.
        v_ctrl_initializer: Initializer for the v control variable.
        trainable: Boolean, if `True` also add variables to the graph
            collection `GraphKeys.TRAINABLE_VARIABLES`
            (see tf.Variable).  (Default: True)

    Input shape:
      Arbitrary. Use the keyword argument `input_shape` (tuple of integers,
                 does not include the samples axis) when using this layer as
                 the first layer in a model.

    Output shape:
        Same shape as input.

    References:
        - [Online Normalization for Training Neural Networks](https://arxiv.org/abs/1905.05894)
    """

    def __init__(self, alpha_fwd=0.999, alpha_bkw=0.99,
                 axis=1, epsilon=1e-5,
                 stream_mu_initializer='zeros', stream_var_initializer='ones',
                 u_ctrl_initializer='zeros', v_ctrl_initializer='zeros',
                 trainable=True, name=None, **kwargs):
        super(Norm, self).__init__(trainable=trainable, name=name, **kwargs)
        # setup mixed precesion
        self.dtype_policy = self._mixed_precision_policy \
            if self._mixed_precision_policy.name == "infer_float32_vars" \
                else self._dtype

        if isinstance(self.dtype_policy, Policy):
            self.mixed_precision = True
            self.fp_type = tf.float32 # full precision
            self.mp_type = tf.float16 # reduced precision
        else:
            self.mixed_precision = False
            self.fp_type = self._dtype if self._dtype else tf.float32 # full precision
            self.mp_type = self.fp_type # reduced precision

        assert axis == 1, 'kernel requires channels_first data_format'

        self.axis = axis
        self.norm_ax = None
        self.epsilon = epsilon

        self.alpha_fwd = alpha_fwd
        self.alpha_bkw = alpha_bkw

        self.stream_mu_initializer = initializers.get(stream_mu_initializer)
        self.stream_var_initializer = initializers.get(stream_var_initializer)
        self.u_ctrl_initializer = initializers.get(u_ctrl_initializer)
        self.v_ctrl_initializer = initializers.get(v_ctrl_initializer)

    def build(self, input_shape):
        """
        Allocation of variables for the normalizer.
        See `call`, `fprop`, and `bprop` for variable usage.
        """
        input_shape = input_shape.as_list()
        ndims = len(input_shape)

        # Convert axis to list and resolve negatives
        if isinstance(self.axis, int):
            self.axis = [self.axis]
        if not isinstance(self.axis, list):
              raise TypeError('axis must be int or list, type given: %s'
                              % type(self.axis))
        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x
        # Validate axes
        for x in self.axis:
            if x < 0 or x >= ndims:
                raise ValueError('Invalid axis: %d' % x)
        if len(self.axis) != len(set(self.axis)):
            raise ValueError('Duplicate axis: %s' % self.axis)

        axis_to_dim = {x: input_shape[x] for x in self.axis}
        for x in axis_to_dim:
            if axis_to_dim[x] is None:
                raise ValueError('Input has undefined `axis` dimension. Input '
                                 'shape: ', input_shape)

        # configure the statistics shape and the axis along which to normalize
        self.norm_ax, stat_shape, self.broadcast_shape = [], [], []
        for idx, ax in enumerate(input_shape):
            if idx in self.axis:
                stat_shape.append(ax)
                self.broadcast_shape.append(ax)
            else:
                if idx:
                    self.norm_ax += [idx - 1]
                    self.broadcast_shape.append(1)

        # streaming normalization statistics
        self.mu = self.add_variable(
            'mu',
            stat_shape,
            initializer=self.stream_mu_initializer,
            dtype=self.fp_type,
            trainable=False,
            experimental_autocast=False,
            use_resource=True,
        )

        self.var = self.add_variable(
            'var',
            stat_shape,
            initializer=self.stream_var_initializer,
            dtype=self.fp_type,
            trainable=False,
            experimental_autocast=False,
            use_resource=True,
        )

        # u and v control variables
        self.u_ctrl = self.add_variable(
            'u_ctrl',
            stat_shape,
            initializer=self.u_ctrl_initializer,
            dtype=self.fp_type,
            trainable=False,
            experimental_autocast=False,
            use_resource=True,
        )

        self.v_ctrl = self.add_variable(
            'v_ctrl',
            stat_shape,
            initializer=self.v_ctrl_initializer,
            dtype=self.fp_type,
            trainable=False,
            experimental_autocast=False,
            use_resource=True,
        )

        self.built = True

    def normalization(self, inputs):
        r"""Applies Normalization (the per feature exponential moving
        average, ema, forward and control process backward part of the Online
        Normalization algorithm) as described in the paper:
        `Online Normalization for Training Neural Networks`.
        This class implements a version of the mathematics below.

        .. math::
            y_t = \frac{x_t - \mu_{t-1}}{\sqrt{\sigma^2_{t-1} + \epsilon}}

            \sigma^2_t = (
                \alpha * \sigma^2_{t-1} +
                \alpha * (1 - \alpha) * (x_t - \mu_{t-1}) ^ 2
            )

            \mu_t = \alpha * \mu_{t-1} + (1 - \alpha) * x_t

        The mean and standard-deviation are estimated per-feature.

        forward is decorated with @tf.custom_gradient and has its backward pass
        defined in backward.

        Arguments
            inputs: input activations

        Returns:
            netout: list: [forward normalized activations,
                           backward function]
        """
        @tf.custom_gradient
        def forward(inputs):
            """
            Function for forward pass.

            Arguments:
                inputs: activations of the current batch

            Returns:
                netout: normalized activations
                backward_wrapper: function handle for custom backward pass
            """
            input_shape = inputs.shape
            inputs = tf.reshape(inputs, [input_shape[0], input_shape[1], -1])
            mu, var = tf.nn.moments(
                tf.cast(inputs, self.fp_type),
                2,
                keep_dims=False
            )
            mean, scale, out_s_mu, out_s_var = online_norm_fwd(
                mu=mu,
                var=var,
                in_s_mu=self.mu,
                in_s_var=self.var,
                afwd=self.alpha_fwd,
                eps=self.epsilon,
                T=self.mp_type
            )
            update_mu = tf.assign(self.mu, out_s_mu, validate_shape=True)
            update_var = tf.assign(self.var ,out_s_var, validate_shape=True)
            with tf.control_dependencies([update_mu, update_var]):
                mean = tf.broadcast_to(tf.expand_dims(mean, -1), inputs.shape)
                scale = tf.expand_dims(scale, -1)
                scale = tf.broadcast_to(scale, inputs.shape)
                outputs = ((inputs - mean) / scale)
                out = tf.reshape(outputs, input_shape)

            def backward(deltas):
                """
                Wrapper for the custom backwards pass using ctrl process
                Note: deltas depends on fprop output

                Arguments:
                    deltas: input deltas from the current batch

                Returns
                    grad_delta: output deltas for inputs
                """
                deltas_shape = deltas.shape
                deltas = tf.reshape(
                    deltas,
                    [deltas_shape[0], deltas_shape[1], -1]
                )
                alpha_bkw = self.alpha_bkw
                out_v, grad_tmp = online_norm_v_ctrl(
                    grad_out=deltas,
                    out=outputs,
                    in_v=self.v_ctrl,
                    abkw=alpha_bkw,
                )

                grad_tmp = grad_tmp / scale
                mu_delta = tf.reduce_mean(tf.cast(grad_tmp, tf.float32), 2)
                out_u, d_u = online_norm_u_ctrl(
                    mu_delta=mu_delta,
                    in_u=self.u_ctrl,
                    abkw=alpha_bkw,
                    T=self.mp_type
                )
                d_u = tf.expand_dims(d_u, -1)
                d_u = tf.broadcast_to(d_u, deltas.shape)
                grad_in = grad_tmp - d_u
                grad_in = tf.reshape(grad_in, deltas_shape)

                update_v = tf.assign(self.v_ctrl, out_v)
                update_u = tf.assign(self.u_ctrl, out_u)

                with tf.control_dependencies(
                    [update_u, update_v, update_mu, update_var]
                ):
                    grad_in = tf.identity(grad_in)
                    return grad_in

            with tf.control_dependencies([update_mu, update_var]):
                return out, backward

        return forward(inputs)

    def call(self, inputs, training=None):
        """
        Call function will be called by __call__
        Arguments:
            inputs: activations into the layer
            training: Boolean to set training or inference mode
        Returns:
            normalized activations with multiplicative scale and additive bias
            corrections
        """
        if training is None:
            training = K.learning_phase()

        # Determine a boolean value for `training`: could be True, False, or None.
        training = tf_utils.constant_value(training)

        input_shape = inputs.get_shape()

        def _bcast(inputs):
            """
            broadcasts tensor for tensor operations with tensor of larger rank
            """
            if inputs is None:
                return None

            bcast_shape = [1] * len(input_shape)
            for a in self.axis:
                bcast_shape[a] = input_shape[a]
            return tf.reshape(inputs, bcast_shape)

        # cast fp16 to fp32
        precise_inputs = tf.cast(inputs, self.mp_type)

        # streaming / control normalization
        if training is not False:
            outputs = self.normalization(precise_inputs)
        else:
            mu = tf.cast(_bcast(self.mu), self.mp_type)
            denom = tf.cast(
                tf.math.sqrt(
                    self.var + self.epsilon
                ),
                self.mp_type
            )
            outputs = (inputs - mu) / _bcast(denom)
        outputs = tf.cast(outputs, self.mp_type)

        outputs = tf.cast(outputs, self.mp_type)

        return outputs



class OnlineNorm(Layer):
    """
    Implementation of the
    [Online Normalization Layer](https://arxiv.org/abs/1905.05894)

    Arguments:
        alpha_fwd: the decay factor to be used in fprop to update statistics.
            Default: 0.999
        alpha_bkw: the decay factor to be used in bprop to control the
            gradients propagating through the network. Default: 0.99
        axis: Integer, the axis that should be normalized. Defualt: 1
            Note kernel only supports 1
        epsilon: a value added to the denominator for numerical stability.
            Default: 1e-5.
        stream_mu_initializer: Initializer for the streaming mean.
        stream_var_initializer: Initializer for the streaming variance.
        u_ctrl_initializer: Initializer for the u control variable.
        v_ctrl_initializer: Initializer for the v control variable.
        center: a boolean value that when set to `True`, this module has
            learnable bias parameters. Default: `True`
        scale: a boolean value that when set to `True`, this module has
            learnable linear parameters. Default: `True`
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
        ecm: a string which defines the error compensation mechanism in OnlineNorm.
            Choice: `ac` (Activation Clamping) | `ls` (Layer Scaling).
        ls_eps: if ecm is `ls`, this is the `ls` eps.
        clamp_val: if ecm is `ac` this is the clamp value.
        trainable: Boolean, if `True` also add variables to the graph
            collection `GraphKeys.TRAINABLE_VARIABLES`
            (see tf.Variable).  (Default: True)

    Input shape:
      Arbitrary. Use the keyword argument `input_shape` (tuple of integers,
                 does not include the samples axis) when using this layer as
                 the first layer in a model.
    Output shape:
        Same shape as input.

    References:
        - [Online Normalization for Training Neural Networks](https://arxiv.org/abs/1905.05894)
    """

    def __init__(self, alpha_fwd=0.999, alpha_bkw=0.99,
                 axis=1, epsilon=1e-5,
                 stream_mu_initializer='zeros', stream_var_initializer='ones',
                 u_ctrl_initializer='zeros', v_ctrl_initializer='zeros',
                 center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None,
                 beta_constraint=None, gamma_constraint=None,
                 ecm='ac', ls_eps=1e-05, clamp_val=5,
                 trainable=True, name=None, **kwargs):
        super(OnlineNorm, self).__init__(trainable=trainable,
                                         name=name, **kwargs)
        # setup mixed precesion
        self.dtype_policy = self._mixed_precision_policy \
            if self._mixed_precision_policy.name == "infer_float32_vars" \
                else self._dtype

        if isinstance(self.dtype_policy, Policy):
            self.mixed_precision = True
            self.fp_type = tf.float32 # full precision
            self.mp_type = tf.float16 # reduced precision
        else:
            self.mixed_precision = False
            self.fp_type = self._dtype if self._dtype else tf.float32 # full precision
            self.mp_type = self.fp_type # reduced precision

        if self.mixed_precision:
            assert beta_regularizer is None, \
                "beta_regularizer not supported for mixed precision"
            assert gamma_regularizer is None, \
                "gamma_regularizer not supported for mixed precision"
            assert beta_constraint is None, \
                "beta_constraint not supported for mixed precision"
            assert gamma_constraint is None, \
                "gamma_constraint not supported for mixed precision"

        self.axis = axis

        self.normalization = Norm(
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            axis=axis,
            epsilon=epsilon,
            stream_mu_initializer=stream_mu_initializer,
            stream_var_initializer=stream_var_initializer,
            u_ctrl_initializer=u_ctrl_initializer,
            v_ctrl_initializer=v_ctrl_initializer,
            trainable=trainable,
            **kwargs
        )

        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

        self.ecm = None
        if ecm == 'ls':
            self.ecm = self.layer_scaling
        elif ecm == 'ac':
            self.ecm = self.activation_clamp
        elif ecm:
            raise ValueError('Invalid input. ecm options: "ls" | "ac" | ""')
        self.ls_eps = ls_eps
        self.clamp_val = clamp_val

    def build(self, input_shape):
        """
        Allocation of variables for the normalizer.
        See `call`, `fprop`, and `bprop` for variable usage.
        """
        input_shape = input_shape.as_list()
        ndims = len(input_shape)

        # Convert axis to list and resolve negatives
        if isinstance(self.axis, int):
            self.axis = [self.axis]
        if not isinstance(self.axis, list):
              raise TypeError('axis must be int or list, type given: %s'
                              % type(self.axis))

        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x

        # Validate axes
        for x in self.axis:
            if x < 0 or x >= ndims:
                raise ValueError('Invalid axis: %d' % x)
        if len(self.axis) != len(set(self.axis)):
            raise ValueError('Duplicate axis: %s' % self.axis)

        param_dtype = self.fp_type

        axis_to_dim = {x: input_shape[x] for x in self.axis}
        for x in axis_to_dim:
            if axis_to_dim[x] is None:
                raise ValueError('Input has undefined `axis` dimension. Input '
                                 'shape: ', input_shape)

        if len(axis_to_dim) == 1:
            # Single axis online norm
            param_shape = (list(axis_to_dim.values())[0],)
        else:
            # Parameter shape is the original shape but 1 in all non-axis dims
            param_shape = [axis_to_dim[i] if i in axis_to_dim
                           else 1 for i in range(ndims)]

        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=param_shape, dtype=param_dtype,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape, dtype=param_dtype,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True
            )
        else:
            self.beta = None

        self.built = True

    def layer_scaling(self, inputs):
        """
        Scale full layer by 2nd moment

        Arguments:
            inputs: input activations

        Returns
            activations scaled by their second moment
        """
        scale = tf.cast(
            tf.reduce_mean(
                tf.cast(inputs * inputs, self.fp_type),
                axis=list(range(len(inputs.get_shape())))[1:],
                keepdims=True
            ),
            self.mp_type
        )

        return inputs * tf.rsqrt(scale + self.ls_eps)

    def activation_clamp(self, inputs):
        """
        Clips the output of CN.

        Arguments:
            inputs: input activations

        Returns
            clamped activations
        """
        return tf.clip_by_value(inputs, -self.clamp_val, self.clamp_val)

    def call(self, inputs, training=None):
        """
        Call function will be called by __call__

        Arguments:
            inputs: activations into the layer
            training: Boolean to set training or inference mode

        Returns:
            normalized activations with multiplicative scale and additive bias
            corrections
        """
        if training is None:
            training = K.learning_phase()

        input_shape = inputs.get_shape()

        def _bcast(inputs):
            """
            broadcasts tensor for tensor operations with tensor of larger rank
            """
            if inputs is None:
                return None

            bcast_shape = [1] * len(input_shape)
            for a in self.axis:
                bcast_shape[a] = input_shape[a]
            return tf.reshape(inputs, bcast_shape)

        # streaming / control normalization
        outputs = self.normalization(inputs, training)

        # scale and bias
        if self.scale:
            outputs *= tf.cast(_bcast(self.gamma), self.mp_type)
        if self.center:
            outputs += tf.cast(_bcast(self.beta), self.mp_type)

        # apply error compensation mechanism
        if self.ecm:
            outputs = self.ecm(outputs)

        outputs = tf.cast(outputs, self.mp_type)

        return outputs


def online_norm(
    inputs,
    training=False,
    alpha_fwd=0.999,
    alpha_bkw=0.99,
    axis=1,
    epsilon=1e-5,
    stream_mu_initializer='zeros',
    stream_var_initializer='ones',
    u_ctrl_initializer='zeros',
    v_ctrl_initializer='zeros',
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    ecm='ac',
    ls_eps=1e-05,
    clamp_val=5,
    trainable=True,
    name=None,
    **kwargs
):
    """
    Functional interface to the Online Normalization Layer defined above

    Arguments:
        inputs: The inputs to the layer.
        training: a boolean value that when set to `True`, the
            Online Normalization layer is in training mode. (Default: False)
        alpha_fwd: the decay factor to be used in fprop to update statistics.
            Default: 0.999
        alpha_bkw: the decay factor to be used in bprop to control the
            gradients propagating through the network. Default: 0.99
        layer_scaling: a boolean determining whether layer scaling is applied
            as the final stage of normalization. Default: `True`
        axis: Integer, the axis that should be normalized. Defualt: 1
            Note kernel only supports 1
        epsilon: a value added to the denominator for numerical stability.
            Default: 1e-5.
        center: a boolean value that when set to `True`, this module has
            learnable bias parameters. Default: `True`
        scale: a boolean value that when set to `True`, this module has
            learnable linear parameters. Default: `True`
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        stream_mu_initializer: Initializer for the streaming mean.
        stream_var_initializer: Initializer for the streaming variance.
        u_ctrl_initializer: Initializer for the u control variable.
        v_ctrl_initializer: Initializer for the v control variable.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
        trainable: Boolean, if `True` also add variables to the graph
            collection `GraphKeys.TRAINABLE_VARIABLES`
            (see tf.Variable). (Default: True)

    Return:
        Normalization Layer output
    """
    layer = OnlineNorm(
        alpha_fwd=alpha_fwd,
        alpha_bkw=alpha_bkw,
        axis=axis,
        epsilon=epsilon,
        stream_mu_initializer=stream_mu_initializer,
        stream_var_initializer=stream_var_initializer,
        u_ctrl_initializer=u_ctrl_initializer,
        v_ctrl_initializer=v_ctrl_initializer,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        ecm=ecm,
        ls_eps=ls_eps,
        clamp_val=clamp_val,
        trainable=trainable,
        name=name,
        **kwargs
    )
    return layer(inputs, training=training)
