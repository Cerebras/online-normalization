"""
Released under BSD 3-Clause License, 
Copyright (c) 2019 Cerebras Systems Inc.
All rights reserved.

TensorFlow Implementation of the Online Normalization Layer
"""

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes, tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints, initializers, regularizers
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.layers import Layer


class BatchOnlineNorm(Layer):
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
            (see tf.Variable). (Default: True)
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
        super(BatchOnlineNorm, self).__init__(trainable=trainable,
                                              name=name, **kwargs)
        self.afwd = alpha_fwd
        self.abkw = alpha_bkw

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

        The math above represents the calculations occurring in the layer. To
        speed up computation with batched training we linearize the computation
        along the batch dimension and use convolutions in place of loops to
        distribute the computation across compute fabric.

        forward is decorated with @tf.custom_gradient and has its backward pass
        defined in backward.

        Arguments
            inputs: input activations

        Returns:
            netout: list: [forward normalized activations,
                           backward function]
        """
        def momentum_stat(stat_prev, stat_curr, stat_stream,
                          momentum, momentum_pow, momentum_batch):
            """
            Helper function for streaming across the batch dimension for a
            momentum system using linear operations.
            Useful for GPU acceleration of streaming control layer.

            Used in mu, var, and u_ctrl updates
            Note: v_ctrl needs separate controller

            Arguments
                stat_prev: previous time steps statistics
                stat_curr: this time steps statistics
                stat_stream: the moving momentum average of the statistics
                momentum: the momentum of the system
                momentum_pow: momentum ** range(b_size - 1, -1, -1)
                momentum_batch: momentum ** b_size

            Returns:
                stream_t1: 1 time step stale estimates of statistics
                           one for each example in batch
                stream_curr: estimates of statistics
                             one for each example in batch

            """

            tmp = tf.concat([stat_prev, stat_curr], axis=0)[1:]

            cm = tf.nn.convolution(tf.expand_dims(tf.transpose(tmp), -1),
                                   momentum_pow, padding='VALID',
                                   data_format="NWC")[:, :, 0]
            cm = tf.transpose(cm)

            stream_curr = stat_stream * momentum_batch + (1.0 - momentum) * cm
            stream_t1 = tf.concat([tf.expand_dims(stat_stream[-1], 0),
                                   stream_curr[:-1]], axis=0)

            return stream_t1, stream_curr

        def conv_alongb_w1(input, b=self.b_size, c=self.ch):
            """
            Helper function to convolve along 2b dimension with a b length
            vector of 1's

            Arguments:
                input: input of shape (b, 2b, c)
                b: b_size
                c: number of features

            Returns
                d: deltas convolved along the 2b dimension with a 1 filter

            """
            c_input = tf.transpose(tf.reshape(tf.transpose(input,
                                                           perm=[1, 0, 2]),
                                              [2 * b, -1]))
            out = tf.nn.conv1d(tf.expand_dims(c_input, 2),
                               tf.constant([1.], shape=[b, 1, 1]),
                               stride=1, padding='VALID', data_format="NWC")
            return tf.transpose(tf.reshape(tf.transpose(tf.squeeze(out)),
                                           [b + 1, b, c]),
                                perm=[1, 0, 2])

        def reshape(input, norm_ax=self.norm_ax):
            if not isinstance(norm_ax, list):
                norm_ax = norm_ax.as_list()

            # take care of channels last
            if norm_ax == [2, 3]:
                return tf.reshape(input, [self.b_size, self.ch, 1, 1])
            # take care of channel last
            if norm_ax == [1, 2]:
                return tf.reshape(input, [self.b_size, 1, 1, self.ch])
            # take care of fc layer
            if norm_ax == []:
                return tf.reshape(input, [self.b_size, self.ch])

        def lin_v_crtl(delta, out, v_p, alpha_p, beta_p,
                       b_size=self.b_size, num_features=self.ch,
                       abkw=self.abkw, norm_ax=self.norm_ax, clip_min=1e-32):
            """
            Helper function to linearize the v controller

            Note:
                - This is originally created / hard-coded for channel_first
                operation
                - There are edge cases where the math breaks down
                i.e. we take a log of alpha, therefore we must clip alpha such
                that alpha > 0. alpha = 1 - (1 - abkw) * (out ** 2)

            Arguments:
                deltas: input deltas to the v controller
                out: the output of the forward pass
                v_p: v controller from the previous time step
                alpha_p: alpha from the previous time step
                beta_p: beta from the previous time step
                b_size: b_size
                num_features: number of features
                abkw: decay factor for the controller
                clip_min: the epsilon by which to clip alpha

            Returns
                grad_delta: output deltas for inputs
                v_new: v control current
                alpha: alpha of this time step to be cached
                beta: beta of this time step to be cached
            """
            # expect 0 << alpha ~<1 so we can move it to log space
            alpha = 1 - (1. - abkw) * tf.reduce_mean(out * out, axis=norm_ax)
            alpha = tf.clip_by_value(alpha, clip_min, 1e32)

            beta = tf.reduce_mean(delta * out, axis=norm_ax)

            alpha2log = tf.log(tf.concat([alpha_p, alpha], 0))

            beta2 = tf.concat([beta_p, beta], 0)

            # create circulant matrix out of alpha2log
            # slice first quadrant of it.
            Acirlog = tf.reshape(tf.tile(alpha2log,
                                         [2 * b_size + 1, 1]),
                                 [2 * b_size, 2 * b_size + 1,
                                  num_features])[1:b_size + 1, :b_size]
            # concatenate wit zeros for conv op
            Acirlog2 = tf.concat([Acirlog,
                                  tf.constant([0.],
                                              shape=[b_size,
                                                     b_size, num_features])],
                                 1)

            # conv along batch dim
            # in log same as prod over the same indexes
            Aconvlog = conv_alongb_w1(Acirlog2, b_size, num_features)

            CD = tf.exp(Aconvlog)
            Bcir = tf.reshape(tf.tile(beta2, [2 * b_size + 1, 1]),
                              [2 * b_size, 2 * b_size + 1,
                               num_features])[1:b_size + 1, :b_size]

            VB = tf.concat([tf.reshape(v_p, [b_size, 1, num_features]),
                            Bcir], 1)

            v_new = tf.reduce_sum(CD * VB, axis=1)

            vp = tf.concat([tf.expand_dims(v_p[-1], 0), v_new[:-1]], 0)

            return ((delta - (1. - abkw) * reshape(vp, norm_ax=norm_ax) * out),
                    v_new, alpha, beta)


        def backward(deltas):
            """
            Wrapper for the custom backwards pass using ctrl process
            Note: deltas depends on fprop output

            Arguments:
                deltas: input deltas from the current batch

            Returns
                grad_delta: output deltas for inputs
            """

            abkw = self.abkw
            with tf.control_dependencies([deltas, self.outputs, self.s]):

                # control with v
                v_ctrl_out = lin_v_crtl(deltas, self.outputs,
                                        self.v_p, self.alpha_p, self.beta_p,
                                        b_size=self.b_size,
                                        num_features=self.ch,
                                        abkw=abkw)

                delta_temp, v_p, alpha_p, beta_p = v_ctrl_out
                # grad_delta, v_p, alpha_p, beta_p = v_ctrl_out

                # scale deltas
                delta_temp_scaled = delta_temp / reshape(self.s,
                                                         norm_ax=self.norm_ax)

                # linearized u controller
                dmean = tf.reduce_mean(delta_temp_scaled,
                                       axis=tuple(self.norm_ax),
                                       keepdims=False)
                _u_ctrl, u_ctrl = momentum_stat(self.u_ctrl_p, dmean,
                                                self.u_ctrl, abkw,
                                                self.abpow, self.abbatch)

                with tf.control_dependencies([_u_ctrl, u_ctrl]):
                    u_update = self.u_ctrl.assign(u_ctrl)
                    u_p_update = self.u_ctrl_p.assign(dmean)

                    v_p_update = self.v_p.assign(v_p)
                    alpha_p_update = self.alpha_p.assign(alpha_p)
                    beta_p_update = self.beta_p.assign(beta_p)

                    grad_delta = delta_temp_scaled - reshape(_u_ctrl,
                                                             norm_ax=self.norm_ax)

                with tf.control_dependencies([u_update, u_p_update, v_p_update,
                                              alpha_p_update, beta_p_update]):
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
            afwd = self.afwd
            mu, var = tf.nn.moments(inputs, self.norm_ax)

            # get instance statistics
            _mu_b, mu_b = momentum_stat(self.mu_p, mu, self.mu,
                                        afwd, self.afpow, self.afbatch)

            var_current = var + afwd * (mu - _mu_b) ** 2
            s, var_b = momentum_stat(self.var_p, var_current, self.var,
                                     afwd, self.afpow, self.afbatch)

            scale = self.s.assign(tf.sqrt(s + self.epsilon))

            with tf.control_dependencies([scale]):
                # perform normalization with previous time steps statistics
                out = tf.nn.batch_normalization(inputs,
                                                reshape(_mu_b,
                                                        norm_ax=self.norm_ax),
                                                reshape(s,
                                                        norm_ax=self.norm_ax),
                                                None, None, self.epsilon)

            out_assign = self.outputs.assign(out)

            with tf.control_dependencies([out_assign]):
                update_mu = self.mu.assign(mu_b)
                update_var = self.var.assign(var_b)

                update_mu_p = self.mu_p.assign(mu)
                update_var_p = self.var_p.assign(var_current)

            with tf.control_dependencies([update_mu, update_var,
                                          update_mu_p, update_var_p]):
                netout = tf.identity(out)
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

        self.ch = input_shape[self.axis]

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

        # Raise parameters of fp16 batch norm to fp32
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
        self.norm_ax, stat_shape, self.broadcast_shape  = [], [], []
        for idx, ax in enumerate(input_shape):
            if idx == 0:
                stat_shape += [self.b_size]
                self.broadcast_shape.append(1)
            elif idx in self.axis:
                stat_shape += [ax]
                self.broadcast_shape.append(ax)
            if idx not in self.axis and idx != 0:
                self.norm_ax += [idx]

        # batch streaming parameters fpass
        self.afbatch = ((self.afwd ** self.b_size) *
                        tf.constant([1.], shape=[self.b_size, self.ch]))
        self.afpow = tf.reshape(self.afwd ** tf.range(self.b_size - 1, -1, -1,
                                                    dtype=tf.float32),
                                (self.b_size, 1, 1))

        # batch streaming parameters fpass
        self.abbatch = ((self.abkw ** self.b_size) *
                        tf.constant([1.], shape=[self.b_size, self.ch]))
        self.abpow = tf.reshape(self.abkw ** tf.range(self.b_size - 1, -1, -1,
                                                    dtype=tf.float32),
                                (self.b_size, 1, 1))

        # streaming normalization statistics
        self.mu = self.add_variable(
            'mu',
            stat_shape,
            initializer=self.stream_mu_initializer,
            dtype=param_dtype,
            trainable=False
        )

        self.var = self.add_variable(
            'var',
            stat_shape,
            initializer=self.stream_var_initializer,
            dtype=param_dtype,
            trainable=False
        )

        # bprop cache variables
        self.s = self.add_variable(
            's',
            stat_shape,
            initializer=self.stream_var_initializer,
            dtype=param_dtype,
            trainable=False
        )

        self.outputs = self.add_variable(
            'outputs',
            [self.b_size] + input_shape[1:],
            initializer=tf.zeros_initializer,
            dtype=param_dtype,
            trainable=False
        )

        # capture previous batch statistics
        self.mu_p = self.add_variable(
            'mu_p',
            stat_shape,
            initializer=self.stream_mu_initializer,
            dtype=param_dtype,
            trainable=False
        )
        self.var_p = self.add_variable(
            'var_p',
            stat_shape,
            initializer=self.stream_var_initializer,
            dtype=param_dtype,
            trainable=False
        )

        # u control variables
        self.u_ctrl = self.add_variable(
            'u_ctrl',
            stat_shape,
            initializer=self.u_ctrl_initializer,
            dtype=param_dtype,
            trainable=False
        )
        # capture stats of d for previous time step needed for u controller
        self.u_ctrl_p = self.add_variable(
            'u_ctrl_p',
            stat_shape,
            initializer=self.u_ctrl_initializer,
            dtype=param_dtype,
            trainable=False
        )

        # v control variables
        self.v_p = self.add_variable(
            'v_p',
            stat_shape,
            initializer=self.v_ctrl_initializer,
            dtype=param_dtype,
            trainable=False
        )

        self.alpha_p = self.add_variable(
            'alpha_p',
            stat_shape,
            initializer=tf.ones_initializer,
            dtype=param_dtype,
            trainable=False
        )

        self.beta_p = self.add_variable(
            'beta_p',
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
                    tf.reshape(self.mu[-1], self.broadcast_shape),
                    tf.reshape(self.var[-1], self.broadcast_shape),
                    None,
                    None,
                    self.epsilon
                )
            )
        else:
            x_norm = tf.nn.batch_normalization(
                precise_inputs,
                tf.reshape(self.mu[-1], self.broadcast_shape),
                tf.reshape(self.var[-1], self.broadcast_shape),
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


def batch_online_norm(inputs,
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
                      b_size=2,
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
    layer = BatchOnlineNorm(alpha_fwd=alpha_fwd,
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
