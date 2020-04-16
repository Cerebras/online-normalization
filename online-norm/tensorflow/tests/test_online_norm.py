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

from tensorflow.keras.mixed_precision.experimental import Policy

tf.logging.set_verbosity(tf.logging.ERROR)


class TestOnlineNorm(unittest.TestCase):


    def template_numerical_comparison_on_vs_np(
        self,
        np_inputs,
        np_grad_out=None,
        axis=1,
        alpha_fwd=0.99,
        alpha_bkw=0.99,
        itrs=2,
        dtype=None,
    ):
        in_shape = np_inputs.shape
        batch_size = in_shape[0]
        NpOnlineNorm = NpOnlineNorm2d if len(in_shape) == 4 else NpOnlineNorm1d
        # Instantiate numpy layer
        np_norm = NpOnlineNorm(
            in_shape[1],
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            affine=False,
            ecm='',
        )

        # Instantiate the tf implementation of online norm layer
        # without batch acceleration
        in_shape = in_shape
        if dtype==None:
            tf_inputs = tf.placeholder(tf.float32, shape=in_shape)
        else:
            tf_inputs = tf.placeholder(tf.float16, shape=in_shape)
        tf_norm = online_norm(
            tf_inputs,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            axis=axis,
            training=True,
            center=False,
            scale=False,
            ecm='',
            dtype=dtype,
        )

        if np_grad_out is not None:
            # set up tf_norm's gradient functionality
            if dtype==None:
                tf_grad_ys = tf.placeholder(tf.float32, shape=in_shape)
            else:
                tf_grad_ys = tf.placeholder(tf.float16, shape=in_shape)
            tf_norm_grad = tf.gradients(
                ys=tf_norm,
                xs=tf_inputs,
                grad_ys=tf_grad_ys
            )
        
        rtol = 1e-4 if dtype==None else 1e-2
        atol = 1e-5 if dtype==None else 1e-3

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Iterate over generated data
            for itr in range(itrs):

                # fprop through numpy Online Norm class
                np_out = np_norm(np_inputs)
                if np_grad_out is not None:
                    # bprop through numpy Online Norm class
                    np_grad_in = np_norm.backward(np_grad_out)

                if np_grad_out is None:
                    # get the output of the tf layer
                    on_tf_out = sess.run(
                        [tf_norm],
                        feed_dict={tf_inputs: np_inputs}
                    )
                    out = np.array(on_tf_out[0])

                    for n in range(batch_size):
                        # numerically compare output
                        err_msg=f'output comparison failed on itr: {itr}, n: {n}'
                        np.testing.assert_allclose(
                            out[n],
                            np_out[n],
                            rtol=rtol, atol=atol, err_msg=err_msg
                        )

                if np_grad_out is not None:
                    # get the deltas of the tf layer
                    grad_dict = {tf_grad_ys: np_grad_out,
                                 tf_inputs: np_inputs}
                    tf_grad_xs = np.array(
                        sess.run(
                            [tf_norm_grad],
                            feed_dict=grad_dict
                        )[0][0]
                    )

                    for n in range(batch_size):
                        # numerically compare deltas
                        err_msg=f'grad comparison failed on itr: {itr}, n: {n}'
                        np.testing.assert_allclose(
                            tf_grad_xs[n],
                            np_grad_in[n],
                            rtol=rtol, atol=atol, err_msg=err_msg
                        )

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

        self.template_numerical_comparison_on_vs_np(
            np_inputs,
            np_grad_out=None,
            axis=1,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            itrs=itrs,
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

        self.template_numerical_comparison_on_vs_np(
            np_inputs,
            np_grad_out=None,
            axis=1,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            itrs=itrs,
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

        self.template_numerical_comparison_on_vs_np(
            np_inputs,
            np_grad_out=None,
            axis=1,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            itrs=itrs,
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

        self.template_numerical_comparison_on_vs_np(
            np_inputs,
            np_grad_out=None,
            axis=1,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            itrs=itrs,
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

        self.template_numerical_comparison_on_vs_np(
            np_inputs,
            np_grad_out=np_grad_out,
            axis=1,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            itrs=itrs,
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

        self.template_numerical_comparison_on_vs_np(
            np_inputs,
            np_grad_out=np_grad_out,
            axis=1,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            itrs=itrs,
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

        self.template_numerical_comparison_on_vs_np(
            np_inputs,
            np_grad_out=np_grad_out,
            axis=1,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            itrs=itrs,
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

        self.template_numerical_comparison_on_vs_np(
            np_inputs,
            np_grad_out=np_grad_out,
            axis=1,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            itrs=itrs,
        )

    def test0411_1d_numerical_comparison_on_fprop_vs_np_batchsize1_mp(
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

        tf.keras.backend.set_floatx('float16')

        self.template_numerical_comparison_on_vs_np(
            np_inputs,
            np_grad_out=None,
            axis=1,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            itrs=itrs,
            dtype=Policy('infer_float32_vars'),
        )

    def test0421_2d_numerical_comparison_on_fprop_vs_np_batchsize1_mp(
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

        tf.keras.backend.set_floatx('float16')

        self.template_numerical_comparison_on_vs_np(
            np_inputs,
            np_grad_out=None,
            axis=1,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            itrs=itrs,
            dtype=Policy('infer_float32_vars'),
        )

    def test041_1d_numerical_comparison_on_fprop_vs_np_mp(
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

        tf.keras.backend.set_floatx('float16')

        self.template_numerical_comparison_on_vs_np(
            np_inputs,
            np_grad_out=None,
            axis=1,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            itrs=itrs,
            dtype=Policy('infer_float32_vars'),
        )

    def test042_2d_numerical_comparison_on_fprop_vs_np_mp(
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

        tf.keras.backend.set_floatx('float16')

        self.template_numerical_comparison_on_vs_np(
            np_inputs,
            np_grad_out=None,
            axis=1,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            itrs=itrs,
            dtype=Policy('infer_float32_vars'),
        )

    def test0511_1d_numerical_comparison_on_vs_np_batchsize1_mp(
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

        tf.keras.backend.set_floatx('float16')

        self.template_numerical_comparison_on_vs_np(
            np_inputs,
            np_grad_out=np_grad_out,
            axis=1,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            itrs=itrs,
            dtype=Policy('infer_float32_vars'),
        )

    def test0521_2d_numerical_comparison_on_vs_np_batchsize1_mp(
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

        tf.keras.backend.set_floatx('float16')

        self.template_numerical_comparison_on_vs_np(
            np_inputs,
            np_grad_out=np_grad_out,
            axis=1,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            itrs=itrs,
            dtype=Policy('infer_float32_vars'),
        )

    def test051_1d_numerical_comparison_on_vs_np_mp(
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

        tf.keras.backend.set_floatx('float16')

        self.template_numerical_comparison_on_vs_np(
            np_inputs,
            np_grad_out=np_grad_out,
            axis=1,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            itrs=itrs,
            dtype=Policy('infer_float32_vars'),
        )

    def test052_2d_numerical_comparison_on_vs_np_mp(
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

        tf.keras.backend.set_floatx('float16')

        self.template_numerical_comparison_on_vs_np(
            np_inputs,
            np_grad_out=np_grad_out,
            axis=1,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            itrs=itrs,
            dtype=Policy('infer_float32_vars'),
        )


if __name__ == '__main__':
    unittest.main()
