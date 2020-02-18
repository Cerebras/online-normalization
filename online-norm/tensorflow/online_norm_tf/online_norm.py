"""
Released under BSD 3-Clause License, 
Copyright (c) 2019 Cerebras Systems Inc.
All rights reserved.

TensorFlow Implementation of the Online Normalization Layer
"""

import tensorflow as tf
from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.layers import base
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.layers import Layer


class OnlineNorm(base.Layer):
    """
    Implementation of the 
    [Online Normalization Algorithm](https://arxiv.org/abs/1905.05894) 

    Note:
        Implemented with custom gradients, using the @tf.custom_gradient
        decorator which requires tf.__version__ >= 1.7

    Arguments:
        alpha_fwd: the decay factor to be used in fprop to update statistics.
            Default: 0.999
        alpha_bkw: the decay factor to be used in bprop to control the
            gradients propagating through the network. Default: 0.99
        layer_scaling: a boolean determining whether layer scaling is applied
            as the final stage of normalization. Default: `True`
        axis: Integer, the axis that should be normalized (typically the
            features axis). For instance, after a `Conv2D` layer with
            `data_format="channels_first"`, set `axis=1` in
            `OnlineNormalization`. Default: -1
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
            (see tf.Variable).  (Default: True)
        b_size: batch size which is being trained. (Default: 1)

    Input shape:
      Arbitrary. Use the keyword argument `input_shape` (tuple of integers,
                 does not include the samples axis) when using this layer as
                 the first layer in a model.
    Output shape:
        Same shape as input.

    References:
        - [Online Normalization for Training Neural Networks](https://arxiv.org/abs/1905.05894)
    """

    def __init__(self, alpha_fwd=0.999, alpha_bkw=0.99, layer_scaling=True,
                 axis=-1, epsilon=1e-5, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 stream_mu_initializer='zeros', stream_var_initializer='ones',
                 u_ctrl_initializer='zeros', v_ctrl_initializer='zeros',
                 beta_regularizer=None, gamma_regularizer=None,
                 beta_constraint=None, gamma_constraint=None,
                 trainable=True, b_size=1, name=None, **kwargs):
        super(OnlineNorm, self).__init__(trainable=trainable,
                                         name=name, **kwargs)
        self.alpha_fwd = alpha_fwd
        self.alpha_bkw = alpha_bkw

        self.ls = layer_scaling

        self.axis = axis[:] if isinstance(axis, list) else axis

        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

        self.stream_mu_initializer = initializers.get(stream_mu_initializer)
        self.stream_var_initializer = initializers.get(stream_var_initializer)
        self.u_ctrl_initializer = initializers.get(u_ctrl_initializer)
        self.v_ctrl_initializer = initializers.get(v_ctrl_initializer)

        self.b_size = b_size
        self.norm_ax = None

    def control_normalization(self, inputs):
        r"""Applies Control Normalization (the per feature exponential moving
        average, ema, forward and control process backward part of the Online
        Normalization algorithm) as described in the paper:
        `Online Normalization for Training Neural Networks`.
        This class implements a version of the mathematics below.

        .. math::
            y_t = \frac{x_t - \mu_{t-1}}{\sqrt{\sigma^2_{t-1} + \epsilon}}

            \sigma^2_t = \alpha * \sigma^2_{t-1} + \alpha * (1 - \alpha) * (x_t - \mu_{t-1}) ^ 2
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
        def backward(deltas):
            """
            Wrapper for the custom backwards pass using ctrl process
            Note: deltas depends on fprop output

            Arguments:
                deltas: input deltas from the current batch

            Returns
                grad_delta: output deltas for inputs
            """

            alpha_bkw = self.alpha_bkw
            with tf.control_dependencies([deltas, self.outputs, self.s]):
                # control with v
                delta_temp = deltas
                delta_temp -= tf.reshape(
                    self.v_ctrl, self.broadcast_shape
                ) * self.outputs * (1 - alpha_bkw)

                # update v control variables
                # update the estimate of v controller, controlling
                # orthogonality to normalizer output
                update_v = tf.assign_add(
                    self.v_ctrl,
                    tf.reduce_mean(
                        delta_temp * self.outputs,
                        axis=self.norm_ax,
                        keepdims=False
                    )
                )
                # scale deltas
                delta_temp_scaled = delta_temp / tf.reshape(self.s, self.broadcast_shape)

                # control with u
                grad_delta = delta_temp_scaled - (1 - alpha_bkw) * tf.reshape(self.u_ctrl, self.broadcast_shape)

                # update u control variables
                # update the estimate u controller which controls
                # orthogonality to the 1 vector
                update_u = tf.assign_add(
                    self.u_ctrl,
                    tf.reduce_mean(
                        grad_delta,
                        axis=self.norm_ax,
                        keepdims=False
                    )
                )

            with tf.control_dependencies([update_u, update_v,
                                          grad_delta]):
                grad_delta = tf.identity(grad_delta)
                return grad_delta

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
            alpha = self.alpha_fwd

            scale = tf.assign(
                self.s,
                tf.sqrt(self.var + self.epsilon)
            )

            with tf.control_dependencies([scale]):
                # perform normalization with previous time steps statistics
                outputs = tf.nn.batch_normalization(
                    inputs,
                    tf.reshape(self.mu, self.broadcast_shape),
                    tf.reshape(self.var, self.broadcast_shape),
                    None,
                    None,
                    self.epsilon
                )

                out_assign = tf.assign(self.outputs, outputs)

            # compute batch statistics
            mu_bn, var_bn = tf.nn.moments(inputs, self.norm_ax, keep_dims=False)

            with tf.control_dependencies([out_assign, mu_bn, var_bn]):
                # get the new mean and variances
                new_mu = self.mu + (1 - alpha) * (mu_bn - self.mu)
                new_var = (alpha * self.var + (1 - alpha) * var_bn +
                           (alpha * (1 - alpha) * tf.square(mu_bn - self.mu)))

            # update the mean and variance
            with tf.control_dependencies([outputs]):
                update_mu = tf.assign(self.mu, new_mu, validate_shape=True)
                update_var = tf.assign(self.var, new_var, validate_shape=True)

            with tf.control_dependencies([update_mu, update_var]):
                netout = tf.identity(outputs)

                # choose back prop algorithm (single or dual stage)
                return netout, backward

        return forward(inputs)

    def layer_scaling(self, inputs):
        """
        Scale full layer by 2nd moment

        Arguments:
            inputs: input activations

        Returns
            activations scaled by their second moment
        """
        scale = tf.reduce_mean(inputs * inputs,
                               axis=list(range(len(inputs.get_shape())))[1:],
                               keepdims=True)
        return inputs * tf.rsqrt(scale + self.epsilon)

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

        # Raise parameters of fp16 norm to fp32
        # something which tf's BN layer builder does
        if self.dtype == dtypes.float16 or self.dtype == dtypes.bfloat16:
            param_dtype = dtypes.float32
        else:
            param_dtype = self.dtype or dtypes.float32

        axis_to_dim = {x: input_shape[x] for x in self.axis}
        for x in axis_to_dim:
            if axis_to_dim[x] is None:
                raise ValueError('Input has undefined `axis` dimension. Input '
                                 'shape: ', input_shape)
        self.input_spec = InputSpec(ndim=ndims, axes=axis_to_dim)

        if len(axis_to_dim) == 1:
            # Single axis online norm
            param_shape = (list(axis_to_dim.values())[0],)
        else:
            # Parameter shape is the original shape but 1 in all non-axis dims
            param_shape = [axis_to_dim[i] if i in axis_to_dim
                           else 1 for i in range(ndims)]

        if self.scale:
            self.gamma = self.add_weight(name='gamma',
                                         shape=param_shape, dtype=param_dtype,
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint,
                                         trainable=True)
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(name='beta',
                                        shape=param_shape, dtype=param_dtype,
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint,
                                        trainable=True)
        else:
            self.beta = None

        # configure the statistics shape and the axis along which to normalize
        self.norm_ax, stat_shape, self.broadcast_shape = [], [], []
        for idx, ax in enumerate(input_shape):
            if idx in self.axis:
                stat_shape.append(ax)
                self.broadcast_shape.append(ax)
            else:
                self.broadcast_shape.append(1)
            if idx not in self.axis:
                self.norm_ax += [idx]

        # streaming normalization statistics
        self.mu = self.add_weight(
            'mu',
            stat_shape,
            initializer=self.stream_mu_initializer,
            dtype=param_dtype,
            trainable=False
        )

        self.var = self.add_weight(
            'var',
            stat_shape,
            initializer=self.stream_var_initializer,
            dtype=param_dtype,
            trainable=False
        )

        # bprop cache variables
        self.s = self.add_weight(
            's',
            stat_shape,
            initializer=self.stream_var_initializer,
            dtype=param_dtype,
            trainable=False
        )

        self.outputs = self.add_weight(
            'outputs',
            [self.b_size] + input_shape[1:],
            initializer=tf.zeros_initializer,
            dtype=param_dtype,
            trainable=False
        )

        # u and v control variables
        self.u_ctrl = self.add_weight(
            'u_ctrl',
            stat_shape,
            initializer=self.u_ctrl_initializer,
            dtype=param_dtype,
            trainable=False
        )

        self.v_ctrl = self.add_weight(
            'v_ctrl',
            stat_shape,
            initializer=self.v_ctrl_initializer,
            dtype=param_dtype,
            trainable=False
        )

        self.built = True

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
        original_training_value = training
        if training is None:
            training = K.learning_phase()

        # Determine a boolean value for `training`: could be True, False, or None.
        training_value = tf_utils.constant_value(training)

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

        mixed_precision = (inputs.dtype == dtypes.float16 or inputs.dtype == dtypes.bfloat16)

        # cast fp16 to fp32
        precise_inputs = inputs
        if mixed_precision:
            precise_inputs = math_ops.cast(inputs, dtypes.float32)

        # streaming / control normalization
        if training_value is not False:
            x_norm = tf_utils.smart_cond(
                training,
                lambda: self.control_normalization(precise_inputs),
                lambda: tf.nn.batch_normalization(
                    precise_inputs,
                    tf.reshape(self.mu, self.broadcast_shape),
                    tf.reshape(self.var, self.broadcast_shape),
                    None,
                    None,
                    self.epsilon
                )
            )
        else:
            x_norm = tf.nn.batch_normalization(
                precise_inputs,
                tf.reshape(self.mu, self.broadcast_shape),
                tf.reshape(self.var, self.broadcast_shape),
                None,
                None,
                self.epsilon
            )

        # scale and bias
        x_scaled = x_norm * _bcast(self.gamma) if self.scale else x_norm
        x_bias = x_scaled + _bcast(self.beta) if self.center else x_scaled

        outputs = self.layer_scaling(x_bias) if self.ls else x_bias

        # if needed, cast back to fp16
        if mixed_precision:
            outputs = math_ops.cast(outputs, inputs.dtype)

        return outputs


def online_norm(inputs,
                training=False,
                alpha_fwd=0.999,
                alpha_bkw=0.99,
                layer_scaling=True,
                axis=-1,
                epsilon=1e-5,
                center=True,
                scale=True,
                beta_initializer='zeros',
                gamma_initializer='ones',
                stream_mu_initializer='zeros',
                stream_var_initializer='ones',
                u_ctrl_initializer='zeros',
                v_ctrl_initializer='zeros',
                beta_regularizer=None,
                gamma_regularizer=None,
                beta_constraint=None,
                gamma_constraint=None,
                trainable=True,
                b_size=1,
                name=None, **kwargs):
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
        axis: Integer, the axis that should be normalized (typically the
            features axis). For instance, after a `Conv2D` layer with
            `data_format="channels_first"`, set `axis=1` in
            `OnlineNormalization`. Default: -1
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
        b_size: batch size which is being trained. (Default: 1)

    Return:
        Normalization Layer output
    """
    layer = OnlineNorm(alpha_fwd=alpha_fwd,
                       alpha_bkw=alpha_bkw,
                       layer_scaling=layer_scaling,
                       axis=axis, epsilon=epsilon,
                       center=center, scale=scale,
                       beta_initializer=beta_initializer,
                       gamma_initializer=gamma_initializer,
                       stream_mu_initializer=stream_mu_initializer,
                       stream_var_initializer=stream_var_initializer,
                       u_ctrl_initializer=u_ctrl_initializer,
                       v_ctrl_initializer=v_ctrl_initializer,
                       beta_regularizer=beta_regularizer,
                       gamma_regularizer=gamma_regularizer,
                       beta_constraint=beta_constraint,
                       gamma_constraint=gamma_constraint,
                       trainable=trainable,
                       b_size=b_size,
                       name=name,
                       **kwargs)
    return layer.apply(inputs, training=training)
