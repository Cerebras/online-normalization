# -*- coding: utf-8 -*-
"""
Released under BSD 3-Clause License,
Copyright (c) 2019 Cerebras Systems Inc.
All rights reserved.

This module tests the Online Normalization module
"""
import logging
import unittest
import numpy as np
import torch

from online_norm_pytorch import OnlineNorm1d
from .np_online_norm_1d import NpOnlineNorm1d

logger = logging.getLogger(__name__)


class TestOnlineNorm1D(unittest.TestCase):
    def test010_nuerical_comparison(
        self,
        batch_size=32,
        num_features=256,
        alpha_fwd=0.99,
        alpha_bkw=0.9,
        itrs=8,
    ):
        """
        numerical comparison of online norm pytorch (with cuda kernel) and numpy
        """
        if not torch.cuda.is_available():
            self.skipTest('CUDA kernel not tested')
            
        device = torch.device('cuda')  # device object representing GPU

        # create inputs
        input = torch.randn(batch_size, num_features) + .25
        np_inputs = input.clone().detach().numpy()
        inputs = input.clone().detach().to(device).requires_grad_(True)
        # instantiate gradient at the output
        grad_out = torch.randn(batch_size, num_features) + .125
        np_grad_out = grad_out.clone().detach().numpy()
        grad_out = grad_out.clone().detach().to(device).requires_grad_(True)

        # instantiate layer
        norm = OnlineNorm1d(
            num_features,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            ecm='',
            affine=False
        ).to(device)

        np_norm = NpOnlineNorm1d(
            num_features,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            affine=False,
            ecm=''
        )

        for itr in range(itrs):
            # fprop through Linearized Online Norm class
            out = norm(inputs)
            # bprop through Linearized Online Norm class
            out.backward(grad_out)
            # fprop through Looping Online Norm class
            np_out = np_norm(np_inputs)
            # bprop through Looping Online Norm class
            np_grad_in = np_norm.backward(np_grad_out)

            # numerically compare output
            err_msg=f'output comparison failed on itr: {itr}'
            np.testing.assert_allclose(
                out.detach().cpu().numpy(),
                np_out,
                rtol=1e-5, atol=1e-6, err_msg=err_msg
            )
            # numerically grad_in
            err_msg=f'grad_in comparison failed on itr: {itr}'
            np.testing.assert_allclose(
                inputs.grad.detach().cpu().numpy(),
                np_grad_in,
                rtol=1e-5, atol=1e-6, err_msg=err_msg
            )

            inputs.grad.zero_()

        logger.info(
            'Algorithm implemented using cuda numerically matches '
            'numpy implementation'
        )

    def test020_nuerical_comparison(
        self,
        batch_size=32,
        num_features=256,
        alpha_fwd=0.99,
        alpha_bkw=0.9,
        itrs=8,
    ):
        """
        numerical comparison of online norm pytorch (c++ cpu implementation)
        and numpy
        """
        device = torch.device('cpu')  # device object representing CPU

        # create inputs
        input = torch.randn(batch_size, num_features) + .25
        np_inputs = input.clone().detach().numpy()
        inputs = input.clone().detach().to(device).requires_grad_(True)
        # instantiate gradient at the output
        grad_out = torch.randn(batch_size, num_features) + .125
        np_grad_out = grad_out.clone().detach().numpy()
        grad_out = grad_out.clone().detach().to(device).requires_grad_(True)

        # instantiate layer
        norm = OnlineNorm1d(
            num_features,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            ecm='',
            affine=False
        ).to(device)

        np_norm = NpOnlineNorm1d(
            num_features,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            affine=False,
            ecm=''
        )

        for itr in range(itrs):
            # fprop through Linearized Online Norm class
            out = norm(inputs)
            # bprop through Linearized Online Norm class
            out.backward(grad_out)
            # fprop through Looping Online Norm class
            np_out = np_norm(np_inputs)
            # bprop through Looping Online Norm class
            np_grad_in = np_norm.backward(np_grad_out)

            # numerically compare output
            err_msg=f'output comparison failed on itr: {itr}'
            np.testing.assert_allclose(
                out.detach().cpu().numpy(),
                np_out,
                rtol=1e-5, atol=1e-6, err_msg=err_msg
            )
            # numerically grad_in
            err_msg=f'grad_in comparison failed on itr: {itr}'
            np.testing.assert_allclose(
                inputs.grad.detach().cpu().numpy(),
                np_grad_in,
                rtol=1e-5, atol=1e-6, err_msg=err_msg
            )

            inputs.grad.zero_()

        logger.info(
            'Algorithm implemented using cpp numerically matches '
            'numpy implementation'
        )

    def test030_mixedprecision(
        self,
        batch_size=32,
        num_features=256,
        alpha_fwd=0.99,
        alpha_bkw=0.9,
    ):
        """
        instantiate mixed / run precision layer
        """
        if not torch.cuda.is_available():
            self.skipTest('Mixed Precision not implemented on CPU in PyTorch')
        
        device = torch.device('cuda')  # device object representing GPU

        # create half precision inputs
        inputs = (
            torch.randn(
                batch_size,
                num_features,
                requires_grad=True,
                device=device
            ) + .25
        ).half()

        # instantiate layer
        norm = OnlineNorm1d(
            num_features,
            alpha_fwd=alpha_fwd,
            alpha_bkw=alpha_bkw,
            ecm='ac'
        ).to(device)

        out = norm(inputs)          # norm fwd
        torch.cuda.synchronize()
        out.sum().backward()        # compute psudo loss and norm bwd


if __name__ == '__main__':
    unittest.main()
