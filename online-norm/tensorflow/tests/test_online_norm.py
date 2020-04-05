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
        batch_size=32,
        num_features=256,
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
        batch_size=32,
        num_features=256,
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
        num_features=256,
        alpha_fwd=0.99,
        alpha_bkw=0.9,
        itrs=32,
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
        num_features=256,
        height=45,
        width=64,
        alpha_fwd=0.99,
        alpha_bkw=0.9,
        itrs=32,
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
        batch_size=32,
        num_features=256,
        alpha_fwd=0.99,
        alpha_bkw=0.9,
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
        batch_size=32,
        num_features=256,
        height=45,
        width=64,
        alpha_fwd=0.99,
        alpha_bkw=0.9,
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
        num_features=256,
        alpha_fwd=0.99,
        alpha_bkw=0.9,
        itrs=32,
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
        num_features=256,
        height=45,
        width=64,
        alpha_fwd=0.99,
        alpha_bkw=0.9,
        itrs=32,
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
        batch_size=32,
        num_features=256,
        alpha_fwd=0.99,
        alpha_bkw=0.9,
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
        batch_size=32,
        num_features=256,
        height=45,
        width=64,
        alpha_fwd=0.99,
        alpha_bkw=0.9,
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
        batch_size=32,
        num_features=256,
        alpha_fwd=0.99,
        alpha_bkw=0.9,
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
        batch_size=32,
        num_features=256,
        height=45,
        width=64,
        alpha_fwd=0.99,
        alpha_bkw=0.9,
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

    # def test051_1d_numerical_comparison_onbatched_vs_np(
    #     self,
    #     batch_size=32,
    #     num_features=256,
    #     alpha_fwd=0.99,
    #     alpha_bkw=0.9,
    #     itrs=2,
    # ):
    #     """
    #     Test ON Batched Layer against ON for 1d inputs
    #     """
    #     # create inputs
    #     np_inputs = np.random.randn(batch_size, num_features) + .25
    #     # instantiate gradient at the output
    #     np_grad_out = np.random.randn(batch_size, num_features) + .125

    #     np_norm = NpOnlineNorm1d(
    #         num_features,
    #         alpha_fwd=alpha_fwd,
    #         alpha_bkw=alpha_bkw,
    #         affine=False,
    #         ecm='',
    #     )

    #     # Instantiate the tf implementation of online norm layer
    #     # without batch acceleration
    #     in_shape = np_inputs.shape
    #     tf_inputs = tf.placeholder(tf.float32, shape=in_shape)
    #     tf_norm = online_norm(
    #         tf_inputs,
    #         alpha_fwd=alpha_fwd,
    #         alpha_bkw=alpha_bkw,
    #         axis=1,
    #         training=True,
    #         center=False,
    #         scale=False,
    #         ecm='',
    #         b_size=batch_size,
    #         batch_acceleration=True,
    #     )

    #     # set up tf_norm's gradient functionality
    #     tf_grad_ys = tf.placeholder(tf.float32, shape=in_shape)
    #     tf_norm_grad = tf.gradients(
    #         ys=tf_norm,
    #         xs=tf_inputs,
    #         grad_ys=tf_grad_ys
    #     )

    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())

    #         # Iterate over generated data
    #         for itr in range(itrs):

    #             # fprop through numpy Online Norm class
    #             np_out = np_norm(np_inputs)
    #             # bprop through numpy Online Norm class
    #             np_grad_in = np_norm.backward(np_grad_out)


    #             # get the output of the tf layer
    #             on_tf_out = sess.run(
    #                 [tf_norm],
    #                 feed_dict={tf_inputs: np_inputs}
    #             )
    #             out = np.array(on_tf_out[0])

    #             # get the deltas of the tf single batch layer
    #             grad_dict = {tf_grad_ys: np_grad_out,
    #                          tf_inputs: np_inputs}
    #             tf_grad_xs = np.array(
    #                 sess.run(
    #                     [tf_norm_grad],
    #                     feed_dict=grad_dict
    #                 )[0][0]
    #             )

    #             # numerically compare output
    #             err_msg=f'output comparison failed on itr: {itr}'
    #             np.testing.assert_allclose(
    #                 out,
    #                 np_out,
    #                 rtol=1e-4, atol=1e-5, err_msg=err_msg
    #             )

    #             # compare deltas using the symmetric isclose
    #             err_msg=f'grad comparison failed on itr: {itr}'
    #             np.testing.assert_allclose(
    #                 tf_grad_xs,
    #                 np_grad_in,
    #                 rtol=1e-4, atol=1e-5, err_msg=err_msg
    #             )

    # def test052_2d_numerical_comparison_onbatched_vs_np(
    #     self,
    #     batch_size=32,
    #     num_features=256,
    #     height=45,
    #     width=64,
    #     alpha_fwd=0.99,
    #     alpha_bkw=0.9,
    #     itrs=2,
    # ):
    #     """
    #     Test the Online Normalization against the numpy implementation
    #     """
    #     # create inputs
    #     np_inputs = np.random.randn(batch_size, num_features, height, width) + .25
    #     # instantiate gradient at the output
    #     np_grad_out = np.random.randn(batch_size, num_features, height, width) + .125

    #     np_norm = NpOnlineNorm2d(
    #         num_features,
    #         alpha_fwd=alpha_fwd,
    #         alpha_bkw=alpha_bkw,
    #         affine=False,
    #         ecm='',
    #     )

    #     # Instantiate the tf implementation of online norm layer
    #     # without batch acceleration
    #     in_shape = np_inputs.shape
    #     tf_inputs = tf.placeholder(tf.float32, shape=in_shape)
    #     tf_norm = online_norm(
    #         tf_inputs,
    #         alpha_fwd=alpha_fwd,
    #         alpha_bkw=alpha_bkw,
    #         axis=1,
    #         training=True,
    #         center=False,
    #         scale=False,
    #         ecm='',
    #         b_size=batch_size,
    #         batch_acceleration=True,
    #     )

    #     # set up tf_norm's gradient functionality
    #     tf_grad_ys = tf.placeholder(tf.float32, shape=in_shape)
    #     tf_norm_grad = tf.gradients(
    #         ys=tf_norm,
    #         xs=tf_inputs,
    #         grad_ys=tf_grad_ys
    #     )

    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())

    #         # Iterate over generated data
    #         for itr in range(itrs):

    #             # fprop through numpy Online Norm class
    #             np_out = np_norm(np_inputs)
    #             # bprop through numpy Online Norm class
    #             np_grad_in = np_norm.backward(np_grad_out)


    #             # get the output of the tf layer
    #             on_tf_out = sess.run(
    #                 [tf_norm],
    #                 feed_dict={tf_inputs: np_inputs}
    #             )
    #             out = np.array(on_tf_out[0])

    #             # get the deltas of the tf single batch layer
    #             grad_dict = {tf_grad_ys: np_grad_out,
    #                          tf_inputs: np_inputs}
    #             tf_grad_xs = np.array(
    #                 sess.run(
    #                     [tf_norm_grad],
    #                     feed_dict=grad_dict
    #                 )[0][0]
    #             )

    #             # numerically compare output
    #             err_msg=f'output comparison failed on itr: {itr}'
    #             np.testing.assert_allclose(
    #                 out,
    #                 np_out,
    #                 rtol=1e-4, atol=1e-5, err_msg=err_msg
    #             )

    #             # compare deltas using the symmetric isclose
    #             err_msg=f'grad comparison failed on itr: {itr}'
    #             np.testing.assert_allclose(
    #                 tf_grad_xs,
    #                 np_grad_in,
    #                 rtol=1e-4, atol=1e-5, err_msg=err_msg
    #             )


#     def test050_bon_vs_on_ch_last(self,
#                                   alpha_fwd=ALPHA_FWD, alpha_bkw=ALPHA_BKW):
#         """
#         Test the Online Normalization Layer's fprop and bprop against the
#         numpy implementation of the layer.

#         NOTE:
#             - layer's mu and var are randomly initialized as well
#             A zero mean unit variance normalization transformation would do
#             nothing therefore the test would be uninformative
#         """
#         # generate the data
#         input_data, deltas_in = gen_data(channel_last=True)

#         # Instantiate the tensorflow implementation of batched on layer
#         in_shape = input_data[0:B_SIZE].shape
#         b_inputs = tf.placeholder(tf.float32, shape=in_shape)
#         b_deltas = tf.placeholder(tf.float32, shape=in_shape)
#         bon_tf = online_norm(b_inputs, alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw,
#                              axis=-1, training=True,
#                              batch_acceleration=True, b_size=B_SIZE, ecm='')
#         # set up on_tf's gradient functionality
#         def grad_func(b_d_in, b_inputs):
#             return tf.gradients(ys=bon_tf, xs=b_inputs, grad_ys=b_d_in)
#         bon_grad = grad_func(b_deltas, b_inputs)

#         grad_in = np.empty(in_shape)
#         # Instantiate tensorflow implementation of the online layer
#         in_shape = input_data[0:1].shape
#         inputs = tf.placeholder(tf.float32, shape=in_shape)
#         deltas = tf.placeholder(tf.float32, shape=in_shape)
#         on_tf = online_norm(inputs, alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw,
#                             axis=-1, training=True,
#                             batch_acceleration=False, b_size=1, ecm='')
#         # set up on_tf's gradient functionality
#         def grad_func(d_in, inputs):
#             return tf.gradients(ys=on_tf, xs=inputs, grad_ys=d_in)
#         on_grad = grad_func(deltas, inputs)

#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())

#             # Iterate over generated data
#             for i in range(len(input_data)):
#                 idx = i % B_SIZE
                
#                 # forward check
#                 if idx == 0:
#                     # get the output of the tf layer
#                     feed_dict = {b_inputs: input_data[i:i + B_SIZE]}
#                     bon_tf_out = sess.run([bon_tf], feed_dict=feed_dict)
#                     bon_tf_out = np.array(bon_tf_out[0])

#                 # get the output of the tf layer
#                 on_tf_out = sess.run([on_tf],
#                                      feed_dict={inputs: input_data[i:i + 1]})
#                 out = np.array(on_tf_out[0])

#                 f_err_str = 'fwd output divergence on itr {}'.format(i)
#                 np.testing.assert_allclose(out, bon_tf_out[idx:idx + 1],
#                                            rtol=RTOL, atol=ATOL,
#                                            err_msg=f_err_str)

#                 # backward check
#                 if idx == 0:
#                     # get the output of the tf layer
#                     grad_dict = {b_deltas: deltas_in[i:i + B_SIZE],
#                                  b_inputs: input_data[i:i + B_SIZE]}

#                     bon_tf_grad_out = np.array(sess.run([bon_grad],
#                                                feed_dict=grad_dict)[0][0])

#                 # get the deltas of the tf single batch layer
#                 grad_dict = {deltas: deltas_in[i:i + 1],
#                              inputs: input_data[i:i + 1]}
#                 grad_in = np.array(sess.run([on_grad],
#                                         feed_dict=grad_dict)[0][0])

#                 b_err_str = 'bkw delta divergence on itr {}'.format(i)
#                 bon_grad_idx = bon_tf_grad_out[idx:idx + 1]
#                 np.testing.assert_allclose(grad_in, bon_grad_idx,
#                                            rtol=RTOL, atol=ATOL,
#                                            err_msg=b_err_str)

if __name__ == '__main__':
    unittest.main()
