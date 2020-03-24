"""
Released under BSD 3-Clause License,
Copyright (c) 2019 Cerebras Systems Inc.
All rights reserved.

Testing for Online Normalization Layer using numpy reference
"""
import unittest
import numpy as np
import tensorflow as tf

from online_norm_tf import online_norm


tf.logging.set_verbosity(tf.logging.ERROR)

# for random generation of test data
WIDTH = 32      # spacial width of data (about 3 x 5)
HEIGHT = 32     # spacial height of data (about 3 x 5)
CHANNELS = 256  # number of channels in generated data
TEST_SIZE = 16  # number of datums in generated data
B_SIZE = 4      # size of batch

TEST_SIZE = B_SIZE * (TEST_SIZE // B_SIZE)

# Hyperparameters for layers
ALPHA_FWD = 0.999
ALPHA_BKW = 0.99

RTOL = 1e-4
ATOL = 1e-5


def gen_data(test_size=TEST_SIZE, channels=CHANNELS,
             width=WIDTH, height=HEIGHT,
             mmean=0, vmean=1, channel_last=False, fc_output=False):
    """
    Generate random data to pass through the layer

    NOTE:
        - The generated data should not be normal, so that the layer can try to
        normalize the data. Therefore the mean of the data is drawn from
        another normal distribution which is centered around 1. This occurs for
        the synthetic activations and synthetic deltas. The variances are
        drawn from a folded normal distribution.

    Return:
        inputs: synthetic input activations
        deltas_in: synthetic deltas for bprop
    """
    if fc_output:
        shape = (test_size, channels)
    else:
        shape = (test_size, channels, width, height)
        if channel_last:
            shape = (test_size, width, height, channels)
    inputs = np.random.normal(loc=0, scale=1, size=shape)

    # Generate random deltas for testing bprop
    deltas_in = np.random.normal(loc=0,
                                 scale=np.abs(vmean * np.random.randn(*shape)),
                                 size=shape)

    return inputs.astype(np.float32), deltas_in.astype(np.float32)


class BatchOnlineNormTest(unittest.TestCase):
    """
    Class for implementing unit tests for the Batch Online Normalization Layer
    """

    def test010_bon_layer_build(self, alpha_fwd=ALPHA_FWD, alpha_bkw=ALPHA_BKW,
                                b_size=B_SIZE, channels=CHANNELS,
                                width=WIDTH, height=HEIGHT):
        """
        Test the Online Normalization Layer builds properly
        """
        # Instantiate the tensorflow implementation of batched online norm layer
        in_shape = (b_size, channels, width, height)
        # statistics and control variables should be reduced along the
        # height, and width dim
        stat_shape = (b_size, channels)

        inputs = tf.placeholder(tf.float32, shape=in_shape)
        bon_tf = online_norm(inputs, alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw,
                             axis=1, training=True, b_size=B_SIZE)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # get pointers to all tf variables
            all_variables = tf.all_variables()

            # get pointers to specific variables needing checking
            mu = [v for v in all_variables if 'mu' in v.name][0]
            var = [v for v in all_variables if 'var' in v.name][0]
            mu_p = [v for v in all_variables if 'mu_p' in v.name][0]
            var_p = [v for v in all_variables if 'var_p' in v.name][0]
            s = [v for v in all_variables if 's' in v.name][0]
            u_ctrl = [v for v in all_variables if 'u_ctrl' in v.name][0]
            u_ctrl_p = [v for v in all_variables if 'u_ctrl_p' in v.name][0]
            v_p = [v for v in all_variables if 'v_p' in v.name][0]
            alpha_p = [v for v in all_variables if 'alpha_p' in v.name][0]
            beta_p = [v for v in all_variables if 'beta_p' in v.name][0]

            # test output shape
            assert_str = 'Output shape should be the same as the input shape'
            assert tuple(bon_tf.shape.as_list()) == in_shape, assert_str

            # test statistics and control variable shape
            assert_str = 'stats trackers and ctrl variables must be \n' \
                         'reduced along the height, and width dim'
            assert tuple(mu.shape.as_list()) == stat_shape, assert_str
            assert tuple(var.shape.as_list()) == stat_shape, assert_str
            assert tuple(mu_p.shape.as_list()) == stat_shape, assert_str
            assert tuple(var_p.shape.as_list()) == stat_shape, assert_str
            assert tuple(s.shape.as_list()) == stat_shape, assert_str
            assert tuple(u_ctrl.shape.as_list()) == stat_shape, assert_str
            assert tuple(u_ctrl_p.shape.as_list()) == stat_shape, assert_str
            assert tuple(v_p.shape.as_list()) == stat_shape, assert_str
            assert tuple(alpha_p.shape.as_list()) == stat_shape, assert_str
            assert tuple(beta_p.shape.as_list()) == stat_shape, assert_str

    def test020_bon_fprop_vs_on(self, alpha_fwd=ALPHA_FWD, alpha_bkw=ALPHA_BKW):
        """
        Test the Batch Online Normalization Layer's forward pass against the
        tf's Online Normalization Layer (b_size=1) implementation of the layer

        NOTE:
            - layer's mu and var are randomly initialized as well
            A zero mean unit variance normalization transformation would do
            nothing therefore the test would be uninformative
        """
        input_data, _ = gen_data()  # generate the data

        # Instantiate the tf implementation of batched online norm layer
        in_shape = input_data[0:B_SIZE].shape

        b_inputs = tf.placeholder(tf.float32, shape=in_shape)
        bon_tf = online_norm(b_inputs, alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw,
                             axis=1, training=True, b_size=B_SIZE)

        # Instantiate tf implementation of the online layer
        in_shape = input_data[0:1].shape
        inputs = tf.placeholder(tf.float32, shape=in_shape)
        on_tf = online_norm(inputs, alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw,
                            axis=1, training=True)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Iterate over generated data
            for i in range(len(input_data)):
                idx = i % B_SIZE

                if idx == 0:
                    # get the output of the tf version of the layer
                    feed_dict = {b_inputs: input_data[i:i + B_SIZE]}
                    bon_tf_out = sess.run([bon_tf], feed_dict=feed_dict)
                    bon_tf_out = np.array(bon_tf_out[0])

                # get the output of the tf version of the layer
                on_tf_out = sess.run([on_tf],
                                     feed_dict={inputs: input_data[i:i + 1]})
                out = np.array(on_tf_out[0])

                f_err_str = 'fwd output divergence on itr {}'.format(i)
                np.testing.assert_allclose(out, bon_tf_out[idx:idx + 1],
                                           rtol=RTOL, atol=ATOL,
                                           err_msg=f_err_str)

    def test030_bon_fprop_vs_on_ls(self,
                                   alpha_fwd=ALPHA_FWD, alpha_bkw=ALPHA_BKW):
        """
        Test the Batch Online Normalization Layer's forward pass against the
        tf's Online Normalization Layer (b_size=1) implementation of the layer

        NOTE:
            - layer's mu and var are randomly initialized as well
            A zero mean unit variance normalization transformation would do
            nothing therefore the test would be uninformative
        """
        input_data, _ = gen_data()  # generate the data

        # Instantiate the tf implementation of batched online norm layer
        in_shape = input_data[0:B_SIZE].shape

        b_inputs = tf.placeholder(tf.float32, shape=in_shape)
        bon_tf = online_norm(b_inputs, alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw,
                             axis=1, training=True, b_size=B_SIZE, ecm='ls')

        # Instantiate tf implementation of the online layer
        in_shape = input_data[0:1].shape
        inputs = tf.placeholder(tf.float32, shape=in_shape)
        on_tf = online_norm(inputs, alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw,
                            axis=1, training=True, ecm='ls')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Iterate over generated data
            for i in range(len(input_data)):
                idx = i % B_SIZE

                if idx == 0:
                    # get the output of the tf version of the layer
                    feed_dict = {b_inputs: input_data[i:i + B_SIZE]}
                    bon_tf_out = sess.run([bon_tf], feed_dict=feed_dict)
                    bon_tf_out = np.array(bon_tf_out[0])

                # get the output of the tf version of the layer
                on_tf_out = sess.run([on_tf],
                                     feed_dict={inputs: input_data[i:i + 1]})
                out = np.array(on_tf_out[0])

                f_err_str = 'fwd output divergence on itr {}'.format(i)
                np.testing.assert_allclose(out, bon_tf_out[idx:idx + 1],
                                           rtol=RTOL, atol=ATOL,
                                           err_msg=f_err_str)

    def test040_bon_vs_on(self, alpha_fwd=ALPHA_FWD, alpha_bkw=ALPHA_BKW):
        """
        Test the Online Normalization Layer's fprop and bprop against the
        numpy implementation of the layer.

        NOTE:
            - layer's mu and var are randomly initialized as well
            A zero mean unit variance normalization transformation would do
            nothing therefore the test would be uninformative
        """
        input_data, deltas_in = gen_data()  # generate the data

        # Instantiate the tensorflow implementation of batched on layer
        in_shape = input_data[0:B_SIZE].shape
        b_inputs = tf.placeholder(tf.float32, shape=in_shape)
        b_deltas = tf.placeholder(tf.float32, shape=in_shape)
        bon_tf = online_norm(b_inputs, alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw,
                             axis=1, training=True, b_size=B_SIZE, ecm='')
        # set up on_tf's gradient functionality
        def grad_func(b_d_in, b_inputs):
            return tf.gradients(ys=bon_tf, xs=b_inputs, grad_ys=b_d_in)
        bon_grad = grad_func(b_deltas, b_inputs)

        grad_in = np.empty(in_shape)
        # Instantiate tensorflow implementation of the online layer
        in_shape = input_data[0:1].shape
        inputs = tf.placeholder(tf.float32, shape=in_shape)
        deltas = tf.placeholder(tf.float32, shape=in_shape)
        on_tf = online_norm(inputs, alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw,
                            axis=1, training=True, ecm='')
        # set up on_tf's gradient functionality
        def grad_func(d_in, inputs):
            return tf.gradients(ys=on_tf, xs=inputs, grad_ys=d_in)
        on_grad = grad_func(deltas, inputs)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Iterate over generated data
            for i in range(len(input_data)):
                idx = i % B_SIZE
                
                # forward check
                if idx == 0:
                    # get the output of the tf version of the layer
                    feed_dict = {b_inputs: input_data[i:i + B_SIZE]}
                    bon_tf_out = sess.run([bon_tf], feed_dict=feed_dict)
                    bon_tf_out = np.array(bon_tf_out[0])

                # get the output of the tf version of the layer
                on_tf_out = sess.run([on_tf],
                                     feed_dict={inputs: input_data[i:i + 1]})
                out = np.array(on_tf_out[0])

                f_err_str = 'fwd output divergence on itr {}'.format(i)
                np.testing.assert_allclose(out, bon_tf_out[idx:idx + 1],
                                           rtol=RTOL, atol=ATOL,
                                           err_msg=f_err_str)

                # backward check
                if idx == 0:
                    # get the output of the tf version of the layer
                    grad_dict = {b_deltas: deltas_in[i:i + B_SIZE],
                                 b_inputs: input_data[i:i + B_SIZE]}

                    bon_tf_grad_out = np.array(sess.run([bon_grad],
                                               feed_dict=grad_dict)[0][0])

                # get the deltas of the tf single batch layer
                grad_dict = {deltas: deltas_in[i:i + 1],
                             inputs: input_data[i:i + 1]}
                grad_in = np.array(sess.run([on_grad],
                                        feed_dict=grad_dict)[0][0])

                # compare deltas using the symmetric isclose
                b_err_str = 'bkw delta divergence on itr {}'.format(i)
                bon_grad_idx = bon_tf_grad_out[idx:idx + 1]
                np.testing.assert_allclose(grad_in, bon_grad_idx,
                                           rtol=RTOL, atol=ATOL,
                                           err_msg=b_err_str)

    def test050_bon_vs_on_ch_last(self,
                                  alpha_fwd=ALPHA_FWD, alpha_bkw=ALPHA_BKW):
        """
        Test the Online Normalization Layer's fprop and bprop against the
        numpy implementation of the layer.

        NOTE:
            - layer's mu and var are randomly initialized as well
            A zero mean unit variance normalization transformation would do
            nothing therefore the test would be uninformative
        """
        # generate the data
        input_data, deltas_in = gen_data(channel_last=True)

        # Instantiate the tensorflow implementation of batched on layer
        in_shape = input_data[0:B_SIZE].shape
        b_inputs = tf.placeholder(tf.float32, shape=in_shape)
        b_deltas = tf.placeholder(tf.float32, shape=in_shape)
        bon_tf = online_norm(b_inputs, alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw,
                             axis=-1, training=True, b_size=B_SIZE, ecm='')
        # set up on_tf's gradient functionality
        def grad_func(b_d_in, b_inputs):
            return tf.gradients(ys=bon_tf, xs=b_inputs, grad_ys=b_d_in)
        bon_grad = grad_func(b_deltas, b_inputs)

        grad_in = np.empty(in_shape)
        # Instantiate tensorflow implementation of the online layer
        in_shape = input_data[0:1].shape
        inputs = tf.placeholder(tf.float32, shape=in_shape)
        deltas = tf.placeholder(tf.float32, shape=in_shape)
        on_tf = online_norm(inputs, alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw,
                            axis=-1, training=True, ecm='')
        # set up on_tf's gradient functionality
        def grad_func(d_in, inputs):
            return tf.gradients(ys=on_tf, xs=inputs, grad_ys=d_in)
        on_grad = grad_func(deltas, inputs)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Iterate over generated data
            for i in range(len(input_data)):
                idx = i % B_SIZE
                
                # forward check
                if idx == 0:
                    # get the output of the tf version of the layer
                    feed_dict = {b_inputs: input_data[i:i + B_SIZE]}
                    bon_tf_out = sess.run([bon_tf], feed_dict=feed_dict)
                    bon_tf_out = np.array(bon_tf_out[0])

                # get the output of the tf version of the layer
                on_tf_out = sess.run([on_tf],
                                     feed_dict={inputs: input_data[i:i + 1]})
                out = np.array(on_tf_out[0])

                f_err_str = 'fwd output divergence on itr {}'.format(i)
                np.testing.assert_allclose(out, bon_tf_out[idx:idx + 1],
                                           rtol=RTOL, atol=ATOL,
                                           err_msg=f_err_str)

                # backward check
                if idx == 0:
                    # get the output of the tf version of the layer
                    grad_dict = {b_deltas: deltas_in[i:i + B_SIZE],
                                 b_inputs: input_data[i:i + B_SIZE]}

                    bon_tf_grad_out = np.array(sess.run([bon_grad],
                                               feed_dict=grad_dict)[0][0])

                # get the deltas of the tf single batch layer
                grad_dict = {deltas: deltas_in[i:i + 1],
                             inputs: input_data[i:i + 1]}
                grad_in = np.array(sess.run([on_grad],
                                        feed_dict=grad_dict)[0][0])

                b_err_str = 'bkw delta divergence on itr {}'.format(i)
                bon_grad_idx = bon_tf_grad_out[idx:idx + 1]
                np.testing.assert_allclose(grad_in, bon_grad_idx,
                                           rtol=RTOL, atol=ATOL,
                                           err_msg=b_err_str)

    def test060_bon_vs_on_Dense(self, alpha_fwd=ALPHA_FWD, alpha_bkw=ALPHA_BKW):
        """
        Test the Online Normalization Layer's fprop and bprop

        NOTE:
            - layer's mu and var are randomly initialized as well
            A zero mean unit variance normalization transformation would do
            nothing therefore the test would be uninformative
        """
        # generate the data
        input_data, deltas_in = gen_data(fc_output=True)

        # Instantiate the tensorflow implementation of batched on layer
        in_shape = input_data[0:B_SIZE].shape
        b_inputs = tf.placeholder(tf.float32, shape=in_shape)
        b_deltas = tf.placeholder(tf.float32, shape=in_shape)
        bon_tf = online_norm(b_inputs, alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw,
                             axis=-1, training=True, b_size=B_SIZE, ecm='')
        # set up on_tf's gradient functionality
        def grad_func(b_d_in, b_inputs):
            return tf.gradients(ys=bon_tf, xs=b_inputs, grad_ys=b_d_in)
        bon_grad = grad_func(b_deltas, b_inputs)

        grad_in = np.empty(in_shape)
        # Instantiate tensorflow implementation of the online layer
        in_shape = input_data[0:1].shape
        inputs = tf.placeholder(tf.float32, shape=in_shape)
        deltas = tf.placeholder(tf.float32, shape=in_shape)
        on_tf = online_norm(inputs, alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw,
                            axis=-1, training=True, ecm='')
        # set up on_tf's gradient functionality
        def grad_func(d_in, inputs):
            return tf.gradients(ys=on_tf, xs=inputs, grad_ys=d_in)
        on_grad = grad_func(deltas, inputs)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Iterate over generated data
            for i in range(len(input_data)):
                idx = i % B_SIZE
                
                # forward check
                if idx == 0:
                    # get the output of the tf version of the layer
                    feed_dict = {b_inputs: input_data[i:i + B_SIZE]}
                    bon_tf_out = sess.run([bon_tf], feed_dict=feed_dict)
                    bon_tf_out = np.array(bon_tf_out[0])

                # get the output of the tf version of the layer
                on_tf_out = sess.run([on_tf],
                                     feed_dict={inputs: input_data[i:i + 1]})
                out = np.array(on_tf_out[0])

                f_err_str = 'fwd output divergence on itr {}'.format(i)
                np.testing.assert_allclose(out, bon_tf_out[idx:idx + 1],
                                           rtol=RTOL, atol=ATOL,
                                           err_msg=f_err_str)

                # backward check
                if idx == 0:
                    # get the output of the tf version of the layer
                    grad_dict = {b_deltas: deltas_in[i:i + B_SIZE],
                                 b_inputs: input_data[i:i + B_SIZE]}

                    bon_tf_grad_out = np.array(sess.run([bon_grad],
                                               feed_dict=grad_dict)[0][0])

                # get the deltas of the tf single batch layer
                grad_dict = {deltas: deltas_in[i:i + 1],
                             inputs: input_data[i:i + 1]}
                grad_in = np.array(sess.run([on_grad],
                                        feed_dict=grad_dict)[0][0])

                b_err_str = 'bkw delta divergence on itr {}'.format(i)
                bon_grad_idx = bon_tf_grad_out[idx:idx + 1]
                np.testing.assert_allclose(grad_in, bon_grad_idx,
                                           rtol=RTOL, atol=ATOL,
                                           err_msg=b_err_str)

if __name__ == '__main__':
    unittest.main()
