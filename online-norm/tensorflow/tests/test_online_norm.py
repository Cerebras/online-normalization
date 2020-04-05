"""
Released under BSD 3-Clause License,
Copyright (c) 2019 Cerebras Systems Inc.
All rights reserved.

Testing for Online Normalization Layer using numpy reference
"""
import os
import sys
import unittest
import numpy as np
import tensorflow as tf

from online_norm_tf import online_norm

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
from numpy_on import OnlineNorm1d as NpOnlineNorm1d
from numpy_on import OnlineNorm2d as NpOnlineNorm2d


tf.logging.set_verbosity(tf.logging.ERROR)


class TestOnlineNorm(unittest.TestCase):
    def test011_1d_bon_layer_build(
        self,
        batch_size=8,
        num_features=16,
    ):
        """
        Test the Online Normalization with batch acceleration builds properly for 1d inputs
        """
        # Instantiate the tensorflow implementation of batched online norm layer
        in_shape = (batch_size, num_features)
        # statistics and control variables should be reduced along the
        # height, and width dim
        stat_shape = (batch_size, num_features)

        # Instantiate the tf implementation of online norm layer
        tf_inputs = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm = online_norm(
            tf_inputs,
            axis=1,
            training=True,
            b_size=batch_size,
            batch_acceleration=True,
        )

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
            assert tuple(tf_norm.shape.as_list()) == in_shape, assert_str

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

    def test012_2d_bon_layer_build(
        self,
        batch_size=8,
        num_features=16,
        height=45,
        width=64,
    ):
        """
        Test the Online Normalization with batch acceleration builds properly for 2d inputs
        """
        # Instantiate the tensorflow implementation of batched online norm layer
        in_shape = (batch_size, num_features, height, width)
        # statistics and control variables should be reduced along the
        # height, and width dim
        stat_shape = (batch_size, num_features)

        # Instantiate the tf implementation of online norm layer
        tf_inputs = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm = online_norm(
            tf_inputs,
            axis=1,
            training=True,
            b_size=batch_size,
            batch_acceleration=True,
        )

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
            assert tuple(tf_norm.shape.as_list()) == in_shape, assert_str

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

    def test0211_1d_numerical_comparison_on_fprop_vs_np_batchsize1(
        self,
        batch_size=1,
        num_features=16,
        alpha_fwd=0.99,
        alpha_bkw=0.99,
        itrs=16,
    ):
        """
        Test ON Layer's fprop against numpy implementation for 1d inputs at batch size 1
        """
        # create inputs
        np_inputs = np.random.randn(batch_size, num_features) + .25

        np_norm = NpOnlineNorm1d(
            num_features,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            affine=False,
            ecm='',
        )

        # Instantiate the tf implementation of online norm layer
        # without batch acceleration
        in_shape = np_inputs.shape
        tf_inputs = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm = online_norm(
            tf_inputs,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            axis=1,
            training=True,
            center=False,
            scale=False,
            ecm='',
            b_size=batch_size,
            batch_acceleration=False,
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Iterate over generated data
            for itr in range(itrs):

                # fprop through numpy Online Norm class
                np_out = np_norm(np_inputs)


                # get the output of the tf layer
                on_tf_out = sess.run(
                    [tf_norm],
                    feed_dict={tf_inputs: np_inputs}
                )
                out = np.array(on_tf_out[0])

                # numerically compare output
                err_msg=f'output comparison failed on itr: {itr}'
                np.testing.assert_allclose(
                    out,
                    np_out,
                    rtol=1e-4, atol=1e-5, err_msg=err_msg
                )

    def test0221_2d_numerical_comparison_on_fprop_vs_np_batchsize1(
        self,
        batch_size=1,
        num_features=16,
        height=45,
        width=64,
        alpha_fwd=0.99,
        alpha_bkw=0.99,
        itrs=16,
    ):
        """
        Test ON Layer's fprop against numpy implementation for 2d inputs at batch size 1
        """
        # create inputs
        np_inputs = np.random.randn(batch_size, num_features, height, width) + .25

        np_norm = NpOnlineNorm2d(
            num_features,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            affine=False,
            ecm='',
        )

        # Instantiate the tf implementation of online norm layer
        # without batch acceleration
        in_shape = np_inputs.shape
        tf_inputs = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm = online_norm(
            tf_inputs,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            axis=1,
            training=True,
            center=False,
            scale=False,
            ecm='',
            b_size=batch_size,
            batch_acceleration=False,
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Iterate over generated data
            for itr in range(itrs):

                # fprop through numpy Online Norm class
                np_out = np_norm(np_inputs)


                # get the output of the tf layer
                on_tf_out = sess.run(
                    [tf_norm],
                    feed_dict={tf_inputs: np_inputs}
                )
                out = np.array(on_tf_out[0])

                # numerically compare output
                err_msg=f'output comparison failed on itr: {itr}'
                np.testing.assert_allclose(
                    out,
                    np_out,
                    rtol=1e-4, atol=1e-5, err_msg=err_msg
                )

    def test021_1d_numerical_comparison_on_fprop_vs_np(
        self,
        batch_size=8,
        num_features=16,
        alpha_fwd=0.99,
        alpha_bkw=0.99,
        itrs=2,
    ):
        """
        Test ON Layer's fprop against numpy implementation for 1d inputs
        """
        # create inputs
        np_inputs = np.random.randn(batch_size, num_features) + .25

        np_norm = NpOnlineNorm1d(
            num_features,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            affine=False,
            ecm='',
        )

        # Instantiate the tf implementation of online norm layer
        # without batch acceleration
        in_shape = np_inputs.shape
        tf_inputs = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm = online_norm(
            tf_inputs,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            axis=1,
            training=True,
            center=False,
            scale=False,
            ecm='',
            b_size=batch_size,
            batch_acceleration=False,
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Iterate over generated data
            for itr in range(itrs):

                # fprop through numpy Online Norm class
                np_out = np_norm(np_inputs)


                # get the output of the tf layer
                on_tf_out = sess.run(
                    [tf_norm],
                    feed_dict={tf_inputs: np_inputs}
                )
                out = np.array(on_tf_out[0])

                # numerically compare output
                err_msg=f'output comparison failed on itr: {itr}'
                np.testing.assert_allclose(
                    out,
                    np_out,
                    rtol=1e-4, atol=1e-5, err_msg=err_msg
                )

    def test022_2d_numerical_comparison_on_fprop_vs_np(
        self,
        batch_size=8,
        num_features=16,
        height=45,
        width=64,
        alpha_fwd=0.99,
        alpha_bkw=0.99,
        itrs=2,
    ):
        """
        Test ON Layer's fprop against numpy implementation for 2d inputs
        """
        # create inputs
        np_inputs = np.random.randn(batch_size, num_features, height, width) + .25

        np_norm = NpOnlineNorm2d(
            num_features,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            affine=False,
            ecm='',
        )

        # Instantiate the tf implementation of online norm layer
        # without batch acceleration
        in_shape = np_inputs.shape
        tf_inputs = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm = online_norm(
            tf_inputs,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            axis=1,
            training=True,
            center=False,
            scale=False,
            ecm='',
            b_size=batch_size,
            batch_acceleration=False,
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Iterate over generated data
            for itr in range(itrs):

                # fprop through numpy Online Norm class
                np_out = np_norm(np_inputs)


                # get the output of the tf layer
                on_tf_out = sess.run(
                    [tf_norm],
                    feed_dict={tf_inputs: np_inputs}
                )
                out = np.array(on_tf_out[0])

                # numerically compare output
                err_msg=f'output comparison failed on itr: {itr}'
                np.testing.assert_allclose(
                    out,
                    np_out,
                    rtol=1e-4, atol=1e-5, err_msg=err_msg
                )

    def test0311_1d_numerical_comparison_on_vs_np_batchsize1(
        self,
        batch_size=1,
        num_features=16,
        alpha_fwd=0.99,
        alpha_bkw=0.99,
        itrs=16,
    ):
        """
        Test ON Layer against numpy implementation for 1d inputs at batch size 1
        """
        # create inputs
        np_inputs = np.random.randn(batch_size, num_features) + .25
        # instantiate gradient at the output
        np_grad_out = np.random.randn(batch_size, num_features) + .125

        np_norm = NpOnlineNorm1d(
            num_features,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            affine=False,
            ecm='',
        )

        # Instantiate the tf implementation of online norm layer
        # without batch acceleration
        in_shape = np_inputs.shape
        tf_inputs = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm = online_norm(
            tf_inputs,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            axis=1,
            training=True,
            center=False,
            scale=False,
            ecm='',
            b_size=batch_size,
            batch_acceleration=False,
        )

        # set up tf_norm's gradient functionality
        tf_grad_ys = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm_grad = tf.gradients(
            ys=tf_norm,
            xs=tf_inputs,
            grad_ys=tf_grad_ys
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Iterate over generated data
            for itr in range(itrs):

                # fprop through numpy Online Norm class
                np_out = np_norm(np_inputs)
                # bprop through numpy Online Norm class
                np_grad_in = np_norm.backward(np_grad_out)


                # get the output of the tf layer
                on_tf_out = sess.run(
                    [tf_norm],
                    feed_dict={tf_inputs: np_inputs}
                )
                out = np.array(on_tf_out[0])

                # get the deltas of the tf layer
                grad_dict = {tf_grad_ys: np_grad_out,
                             tf_inputs: np_inputs}
                tf_grad_xs = np.array(
                    sess.run(
                        [tf_norm_grad],
                        feed_dict=grad_dict
                    )[0][0]
                )

                # numerically compare output
                err_msg=f'output comparison failed on itr: {itr}'
                np.testing.assert_allclose(
                    out,
                    np_out,
                    rtol=1e-4, atol=1e-5, err_msg=err_msg
                )

                # numerically compare deltas
                err_msg=f'grad comparison failed on itr: {itr}'
                np.testing.assert_allclose(
                    tf_grad_xs,
                    np_grad_in,
                    rtol=1e-4, atol=1e-5, err_msg=err_msg
                )

    def test0321_2d_numerical_comparison_on_vs_np_batchsize1(
        self,
        batch_size=1,
        num_features=16,
        height=45,
        width=64,
        alpha_fwd=0.99,
        alpha_bkw=0.99,
        itrs=16,
    ):
        """
        Test ON Layer against numpy implementation for 2d inputs at batch size 1
        """
        # create inputs
        np_inputs = np.random.randn(batch_size, num_features, height, width) + .25
        # instantiate gradient at the output
        np_grad_out = np.random.randn(batch_size, num_features, height, width) + .125

        np_norm = NpOnlineNorm2d(
            num_features,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            affine=False,
            ecm='',
        )

        # Instantiate the tf implementation of online norm layer
        # without batch acceleration
        in_shape = np_inputs.shape
        tf_inputs = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm = online_norm(
            tf_inputs,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            axis=1,
            training=True,
            center=False,
            scale=False,
            ecm='',
            b_size=batch_size,
            batch_acceleration=False,
        )

        # set up tf_norm's gradient functionality
        tf_grad_ys = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm_grad = tf.gradients(
            ys=tf_norm,
            xs=tf_inputs,
            grad_ys=tf_grad_ys
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Iterate over generated data
            for itr in range(itrs):

                # fprop through numpy Online Norm class
                np_out = np_norm(np_inputs)
                # bprop through numpy Online Norm class
                np_grad_in = np_norm.backward(np_grad_out)


                # get the output of the tf layer
                on_tf_out = sess.run(
                    [tf_norm],
                    feed_dict={tf_inputs: np_inputs}
                )
                out = np.array(on_tf_out[0])

                # get the deltas of the tf layer
                grad_dict = {tf_grad_ys: np_grad_out,
                             tf_inputs: np_inputs}
                tf_grad_xs = np.array(
                    sess.run(
                        [tf_norm_grad],
                        feed_dict=grad_dict
                    )[0][0]
                )

                # numerically compare output
                err_msg=f'output comparison failed on itr: {itr}'
                np.testing.assert_allclose(
                    out,
                    np_out,
                    rtol=1e-4, atol=1e-5, err_msg=err_msg
                )

                # numerically compare deltas
                err_msg=f'grad comparison failed on itr: {itr}'
                np.testing.assert_allclose(
                    tf_grad_xs,
                    np_grad_in,
                    rtol=1e-4, atol=1e-5, err_msg=err_msg
                )

    def test031_1d_numerical_comparison_on_vs_np(
        self,
        batch_size=8,
        num_features=16,
        alpha_fwd=0.99,
        alpha_bkw=0.99,
        itrs=2,
    ):
        """
        Test ON Layer against numpy implementation for 1d inputs
        """
        # create inputs
        np_inputs = np.random.randn(batch_size, num_features) + .25
        # instantiate gradient at the output
        np_grad_out = np.random.randn(batch_size, num_features) + .125

        np_norm = NpOnlineNorm1d(
            num_features,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            affine=False,
            ecm='',
        )

        # Instantiate the tf implementation of online norm layer
        # without batch acceleration
        in_shape = np_inputs.shape
        tf_inputs = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm = online_norm(
            tf_inputs,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            axis=1,
            training=True,
            center=False,
            scale=False,
            ecm='',
            b_size=batch_size,
            batch_acceleration=False,
        )

        # set up tf_norm's gradient functionality
        tf_grad_ys = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm_grad = tf.gradients(
            ys=tf_norm,
            xs=tf_inputs,
            grad_ys=tf_grad_ys
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Iterate over generated data
            for itr in range(itrs):

                # fprop through numpy Online Norm class
                np_out = np_norm(np_inputs)
                # bprop through numpy Online Norm class
                np_grad_in = np_norm.backward(np_grad_out)


                # get the output of the tf layer
                on_tf_out = sess.run(
                    [tf_norm],
                    feed_dict={tf_inputs: np_inputs}
                )
                out = np.array(on_tf_out[0])

                # get the deltas of the tf layer
                grad_dict = {tf_grad_ys: np_grad_out,
                             tf_inputs: np_inputs}
                tf_grad_xs = np.array(
                    sess.run(
                        [tf_norm_grad],
                        feed_dict=grad_dict
                    )[0][0]
                )

                # numerically compare output
                err_msg=f'output comparison failed on itr: {itr}'
                np.testing.assert_allclose(
                    out,
                    np_out,
                    rtol=1e-4, atol=1e-5, err_msg=err_msg
                )

                # numerically compare deltas
                err_msg=f'grad comparison failed on itr: {itr}'
                np.testing.assert_allclose(
                    tf_grad_xs,
                    np_grad_in,
                    rtol=1e-4, atol=1e-5, err_msg=err_msg
                )

    def test032_2d_numerical_comparison_on_vs_np(
        self,
        batch_size=8,
        num_features=16,
        height=45,
        width=64,
        alpha_fwd=0.99,
        alpha_bkw=0.99,
        itrs=2,
    ):
        """
        Test ON Layer against numpy implementation for 2d inputs
        """
        # create inputs
        np_inputs = np.random.randn(batch_size, num_features, height, width) + .25
        # instantiate gradient at the output
        np_grad_out = np.random.randn(batch_size, num_features, height, width) + .125

        np_norm = NpOnlineNorm2d(
            num_features,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            affine=False,
            ecm='',
        )

        # Instantiate the tf implementation of online norm layer
        # without batch acceleration
        in_shape = np_inputs.shape
        tf_inputs = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm = online_norm(
            tf_inputs,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            axis=1,
            training=True,
            center=False,
            scale=False,
            ecm='',
            b_size=batch_size,
            batch_acceleration=False,
        )

        # set up tf_norm's gradient functionality
        tf_grad_ys = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm_grad = tf.gradients(
            ys=tf_norm,
            xs=tf_inputs,
            grad_ys=tf_grad_ys
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Iterate over generated data
            for itr in range(itrs):

                # fprop through numpy Online Norm class
                np_out = np_norm(np_inputs)
                # bprop through numpy Online Norm class
                np_grad_in = np_norm.backward(np_grad_out)


                # get the output of the tf layer
                on_tf_out = sess.run(
                    [tf_norm],
                    feed_dict={tf_inputs: np_inputs}
                )
                out = np.array(on_tf_out[0])

                # get the deltas of the tf layer
                grad_dict = {tf_grad_ys: np_grad_out,
                             tf_inputs: np_inputs}
                tf_grad_xs = np.array(
                    sess.run(
                        [tf_norm_grad],
                        feed_dict=grad_dict
                    )[0][0]
                )

                # numerically compare output
                err_msg=f'output comparison failed on itr: {itr}'
                np.testing.assert_allclose(
                    out,
                    np_out,
                    rtol=1e-4, atol=1e-5, err_msg=err_msg
                )

                # numerically compare deltas
                err_msg=f'grad comparison failed on itr: {itr}'
                np.testing.assert_allclose(
                    tf_grad_xs,
                    np_grad_in,
                    rtol=1e-4, atol=1e-5, err_msg=err_msg
                )

    def test041_1d_numerical_comparison_onbatched_vs_on(
        self,
        batch_size=8,
        num_features=16,
        alpha_fwd=0.99,
        alpha_bkw=0.99,
        itrs=2,
    ):
        """
        Test ON Batched Layer's fprop against ON for 1d inputs
        """
        # create inputs
        np_inputs = np.random.randn(batch_size, num_features) + .25

        # Instantiate the tf implementation of online norm layer
        # without batch acceleration
        in_shape = np_inputs.shape
        tf_inputs = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm = online_norm(
            tf_inputs,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            axis=1,
            training=True,
            center=False,
            scale=False,
            ecm='',
            b_size=batch_size,
            batch_acceleration=False,
        )

        # Instantiate the batched implementation of online norm layer
        tf_inputs_batched = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm_batched = online_norm(
            tf_inputs_batched,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            axis=1,
            training=True,
            center=False,
            scale=False,
            ecm='',
            b_size=batch_size,
            batch_acceleration=True,
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for itr in range(itrs):

                # get the output of the tf layer
                on_tf_out = sess.run(
                    [tf_norm],
                    feed_dict={tf_inputs: np_inputs}
                )
                out = np.array(on_tf_out[0])


                # get the output of the batched version of the layer
                on_tf_out_batched = sess.run(
                    [tf_norm_batched],
                    feed_dict={tf_inputs_batched: np_inputs}
                )
                out_batched = np.array(on_tf_out_batched[0])

                # numerically compare output
                err_msg=f'output comparison failed on itr: {itr}'
                np.testing.assert_allclose(
                    out,
                    out_batched,
                    rtol=1e-4, atol=1e-5, err_msg=err_msg
                )

    def test042_2d_numerical_comparison_onbatched_fprop_vs_on(
        self,
        batch_size=8,
        num_features=16,
        height=45,
        width=64,
        alpha_fwd=0.99,
        alpha_bkw=0.99,
        itrs=2,
    ):
        """
        Test ON Batched Layer's fprop against ON for 2d inputs
        """
        # create inputs
        np_inputs = np.random.randn(batch_size, num_features, height, width) + .25

        # Instantiate the tf implementation of online norm layer
        # without batch acceleration
        in_shape = np_inputs.shape
        tf_inputs = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm = online_norm(
            tf_inputs,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            axis=1,
            training=True,
            center=False,
            scale=False,
            ecm='',
            b_size=batch_size,
            batch_acceleration=False,
        )

        # Instantiate the batched implementation of online norm layer
        in_shape_batched = np_inputs.shape
        tf_inputs_batched = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm_batched = online_norm(
            tf_inputs_batched,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            axis=1,
            training=True,
            center=False,
            scale=False,
            ecm='',
            b_size=batch_size,
            batch_acceleration=True,
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for itr in range(itrs):

                # get the output of the tf layer
                on_tf_out = sess.run(
                    [tf_norm],
                    feed_dict={tf_inputs: np_inputs}
                )
                out = np.array(on_tf_out[0])


                # get the output of the batched version of the layer
                on_tf_out_batched = sess.run(
                    [tf_norm_batched],
                    feed_dict={tf_inputs_batched: np_inputs}
                )
                out_batched = np.array(on_tf_out_batched[0])

                # numerically compare output
                err_msg=f'output comparison failed on itr: {itr}'
                np.testing.assert_allclose(
                    out,
                    out_batched,
                    rtol=1e-4, atol=1e-5, err_msg=err_msg
                )

    def test051_1d_numerical_comparison_onbatched_vs_on(
        self,
        batch_size=8,
        num_features=16,
        alpha_fwd=0.99,
        alpha_bkw=0.99,
        itrs=2,
    ):
        """
        Test ON Batched Layer against ON for 1d inputs
        """
        # create inputs
        np_inputs = np.random.randn(batch_size, num_features) + .25
        # instantiate gradient at the output
        np_grad_out = np.random.randn(batch_size, num_features) + .125

        # Instantiate the tf implementation of online norm layer
        # without batch acceleration
        in_shape = np_inputs.shape
        tf_inputs = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm = online_norm(
            tf_inputs,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            axis=1,
            training=True,
            center=False,
            scale=False,
            ecm='',
            b_size=batch_size,
            batch_acceleration=False,
        )

        # set up tf_norm's gradient functionality
        tf_grad_ys = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm_grad = tf.gradients(
            ys=tf_norm,
            xs=tf_inputs,
            grad_ys=tf_grad_ys
        )

        # Instantiate the tf implementation of online norm layer
        in_shape_batched = np_inputs.shape
        tf_inputs_batched = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm_batched = online_norm(
            tf_inputs_batched,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            axis=1,
            training=True,
            center=False,
            scale=False,
            ecm='',
            b_size=batch_size,
            batch_acceleration=True,
        )

        # set up tf_norm's gradient functionality
        tf_grad_ys_batched = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm_grad_batched = tf.gradients(
            ys=tf_norm_batched,
            xs=tf_inputs_batched,
            grad_ys=tf_grad_ys_batched
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Iterate over generated data
            for itr in range(itrs):
                # get the output of the tf layer
                on_tf_out = sess.run(
                    [tf_norm],
                    feed_dict={tf_inputs: np_inputs}
                )
                out = np.array(on_tf_out[0])

                # get the deltas of the tf single batch layer
                grad_dict = {tf_grad_ys: np_grad_out,
                             tf_inputs: np_inputs}
                tf_grad_xs = np.array(
                    sess.run(
                        [tf_norm_grad],
                        feed_dict=grad_dict
                    )[0][0]
                )


                # get the output of the tf layer
                on_tf_out_batched = sess.run(
                    [tf_norm_batched],
                    feed_dict={tf_inputs_batched: np_inputs}
                )
                out_batched = np.array(on_tf_out_batched[0])

                # get the deltas of the tf single batch layer
                grad_dict = {tf_grad_ys_batched: np_grad_out,
                             tf_inputs_batched: np_inputs}
                tf_grad_xs_batched = np.array(
                    sess.run(
                        [tf_norm_grad_batched],
                        feed_dict=grad_dict
                    )[0][0]
                )


                # numerically compare output
                err_msg=f'output comparison failed on itr: {itr}'
                np.testing.assert_allclose(
                    out,
                    out_batched,
                    rtol=1e-4, atol=1e-5, err_msg=err_msg
                )

                # compare deltas
                err_msg=f'grad comparison failed on itr: {itr}'
                np.testing.assert_allclose(
                    tf_grad_xs,
                    tf_grad_xs_batched,
                    rtol=1e-4, atol=1e-5, err_msg=err_msg
                )

    def test052_2d_numerical_comparison_onbatched_vs_on(
        self,
        batch_size=8,
        num_features=16,
        height=45,
        width=64,
        alpha_fwd=0.99,
        alpha_bkw=0.99,
        itrs=2,
    ):
        """
        Test ON Batched Layer against ON for 2d inputs
        """
        # create inputs
        np_inputs = np.random.randn(batch_size, num_features, height, width) + .25
        # instantiate gradient at the output
        np_grad_out = np.random.randn(batch_size, num_features, height, width) + .125

        # Instantiate the tf implementation of online norm layer
        # without batch acceleration
        in_shape = np_inputs.shape
        tf_inputs = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm = online_norm(
            tf_inputs,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            axis=1,
            training=True,
            center=False,
            scale=False,
            ecm='',
            b_size=batch_size,
            batch_acceleration=False,
        )

        # set up tf_norm's gradient functionality
        tf_grad_ys = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm_grad = tf.gradients(
            ys=tf_norm,
            xs=tf_inputs,
            grad_ys=tf_grad_ys
        )


        # Instantiate the tf implementation of online norm layer
        in_shape_batched = np_inputs.shape
        tf_inputs_batched = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm_batched = online_norm(
            tf_inputs_batched,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            axis=1,
            training=True,
            center=False,
            scale=False,
            ecm='',
            b_size=batch_size,
            batch_acceleration=True,
        )

        # set up tf_norm's gradient functionality
        tf_grad_ys_batched = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm_grad_batched = tf.gradients(
            ys=tf_norm_batched,
            xs=tf_inputs_batched,
            grad_ys=tf_grad_ys_batched
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Iterate over generated data
            for itr in range(itrs):
                # get the output of the tf layer
                on_tf_out = sess.run(
                    [tf_norm],
                    feed_dict={tf_inputs: np_inputs}
                )
                out = np.array(on_tf_out[0])

                # get the deltas of the tf layer
                grad_dict = {tf_grad_ys: np_grad_out,
                             tf_inputs: np_inputs}
                tf_grad_xs = np.array(
                    sess.run(
                        [tf_norm_grad],
                        feed_dict=grad_dict
                    )[0][0]
                )

                # get the output of the tf batched layer
                on_tf_out_batched = sess.run(
                    [tf_norm_batched],
                    feed_dict={tf_inputs_batched: np_inputs}
                )
                out_batched = np.array(on_tf_out_batched[0])

                # get the deltas of the tf batched layer
                grad_dict_batched = {tf_grad_ys_batched: np_grad_out,
                             tf_inputs_batched: np_inputs}
                tf_grad_xs_batched = np.array(
                    sess.run(
                        [tf_norm_grad_batched],
                        feed_dict=grad_dict_batched
                    )[0][0]
                )

                # numerically compare output
                err_msg=f'output comparison failed on itr: {itr}'
                np.testing.assert_allclose(
                    out,
                    out_batched,
                    rtol=1e-4, atol=1e-5, err_msg=err_msg
                )

                # numerical compare deltas
                err_msg=f'grad comparison failed on itr: {itr}'
                np.testing.assert_allclose(
                    tf_grad_xs,
                    tf_grad_xs_batched,
                    rtol=1e-4, atol=1e-5, err_msg=err_msg
                )

    def test062_2d_numerical_comparison_onbatched_vs_on_channel_last(
        self,
        batch_size=8,
        num_features=16,
        height=45,
        width=64,
        alpha_fwd=0.99,
        alpha_bkw=0.99,
        itrs=2,
    ):
        """
        Test ON Batched Layer against ON for 2d inputs in channel last mode
        """
        # create inputs
        np_inputs = np.random.randn(batch_size, num_features, height, width) + .25
        # instantiate gradient at the output
        np_grad_out = np.random.randn(batch_size, num_features, height, width) + .125

        # Instantiate the tf implementation of online norm layer
        # without batch acceleration
        in_shape = np_inputs.shape
        tf_inputs = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm = online_norm(
            tf_inputs,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            axis=-1,
            training=True,
            center=False,
            scale=False,
            ecm='',
            b_size=batch_size,
            batch_acceleration=False,
        )

        # set up tf_norm's gradient functionality
        tf_grad_ys = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm_grad = tf.gradients(
            ys=tf_norm,
            xs=tf_inputs,
            grad_ys=tf_grad_ys
        )


        # Instantiate the tf implementation of online norm layer
        in_shape_batched = np_inputs.shape
        tf_inputs_batched = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm_batched = online_norm(
            tf_inputs_batched,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            axis=-1,
            training=True,
            center=False,
            scale=False,
            ecm='',
            b_size=batch_size,
            batch_acceleration=True,
        )

        # set up tf_norm's gradient functionality
        tf_grad_ys_batched = tf.placeholder(tf.float32, shape=in_shape)
        tf_norm_grad_batched = tf.gradients(
            ys=tf_norm_batched,
            xs=tf_inputs_batched,
            grad_ys=tf_grad_ys_batched
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Iterate over generated data
            for itr in range(itrs):
                # get the output of the tf layer
                on_tf_out = sess.run(
                    [tf_norm],
                    feed_dict={tf_inputs: np_inputs}
                )
                out = np.array(on_tf_out[0])

                # get the deltas of the tf layer
                grad_dict = {tf_grad_ys: np_grad_out,
                             tf_inputs: np_inputs}
                tf_grad_xs = np.array(
                    sess.run(
                        [tf_norm_grad],
                        feed_dict=grad_dict
                    )[0][0]
                )

                # get the output of the tf batched layer
                on_tf_out_batched = sess.run(
                    [tf_norm_batched],
                    feed_dict={tf_inputs_batched: np_inputs}
                )
                out_batched = np.array(on_tf_out_batched[0])

                # get the deltas of the tf batched layer
                grad_dict_batched = {tf_grad_ys_batched: np_grad_out,
                             tf_inputs_batched: np_inputs}
                tf_grad_xs_batched = np.array(
                    sess.run(
                        [tf_norm_grad_batched],
                        feed_dict=grad_dict_batched
                    )[0][0]
                )

                # numerically compare output
                err_msg=f'output comparison failed on itr: {itr}'
                np.testing.assert_allclose(
                    out,
                    out_batched,
                    rtol=1e-4, atol=1e-5, err_msg=err_msg
                )

                # numerical compare deltas
                err_msg=f'grad comparison failed on itr: {itr}'
                np.testing.assert_allclose(
                    tf_grad_xs,
                    tf_grad_xs_batched,
                    rtol=1e-4, atol=1e-5, err_msg=err_msg
                )


if __name__ == '__main__':
    unittest.main()
