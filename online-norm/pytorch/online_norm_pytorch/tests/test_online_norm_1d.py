# -*- coding: utf-8 -*-
"""
Released under BSD 3-Clause License,
Copyright (c) 2019 Cerebras Systems Inc.
All rights reserved.

This module tests the Online Normalization module
"""
import time
import logging
import unittest
import numpy as np
import torch

from online_norm_pytorch import OnlineNorm1D


class TestOnlineNorm1D(unittest.TestCase):
    """
    This is the test class which implements the Online Normalization module's
    test
    """
    logger = logging.getLogger('test_logger')

    def test010_similarity(self, batch_size=4, dim=1024,
                           alpha_fwd=0.999, alpha_bkw=0.99, eps=1e-05, itrs=4):
        """ numerical similarity for online norm linearized vs loops """
        # instantiate inputs
        input = torch.randn(batch_size, dim)
        input_0 = input.clone().detach().requires_grad_(True)
        input_1 = input.clone().detach().requires_grad_(True)
        # instantiate gradient at the output
        grad_out = torch.randn(batch_size, dim)

        # instantiate Linearized Online Norm class
        onlin = OnlineNorm1D(dim, batch_size,
                             alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw, eps=eps)

        # instantiate Looping Online Norm class
        onloop = OnlineNorm1D(dim,
                              alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw, eps=eps)

        for _ in range(itrs):
            # fprop through Linearized Online Norm class
            y_0 = onlin(input_0)
            # bprop through Linearized Online Norm class
            y_0.backward(grad_out)
            # fprop through Looping Online Norm class
            y_1 = onloop(input_1)
            # bprop through Looping Online Norm class
            y_1.backward(grad_out)

            # numerically compare output
            np.testing.assert_allclose(y_0.detach().numpy(),
                                       y_1.detach().numpy(),
                                       rtol=1e-4, atol=1e-5)
            # numerically grad_in
            np.testing.assert_allclose(input_0.grad.detach().numpy(),
                                       input_1.grad.detach().numpy(),
                                       rtol=1e-4, atol=1e-5)

        self.logger.info('Algorithm implemented using linearization of ops '
                         'numerically matches algorithm implemented with '
                         'loops')

    def test020_speed(self, batch_size=64, dim=1024,
                      alpha_fwd=0.999, alpha_bkw=0.99, eps=1e-05, epoch=100):
        """
        Speed test online norm linearized vs loops
        Note: this test doesn't check anything it helps the user choose a
            algorithm configuration based on use case.
        """
        input = torch.randn(batch_size, dim)

        # instantiate Linearized Online Norm class
        onlin = OnlineNorm1D(dim, batch_size,
                             alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw, eps=eps)

        # time lin algo
        forward = 0
        backward = 0
        for _ in range(epoch):
            start = time.time()
            # fprop through lin algo
            out = onlin(input)
            forward += time.time() - start

            start = time.time()
            # bprop through lin algo
            out.sum().backward()
            backward += time.time() - start

        self.logger.info(f'Linearized Control Normalization Speed Test: '
                         f'Forward {forward * 1e6/1e5:.3f} us | '
                         f'Backward {backward * 1e6/1e5:.3f} us | '
                         f'Total {(forward + backward) * 1e6/1e5:.3f} us')

        # Speed test online norm
        # instantiate Looping Online Norm class
        onloop = OnlineNorm1D(dim,
                              alpha_fwd=alpha_fwd, alpha_bkw=alpha_bkw, eps=eps)

        # time loop algo
        forward = 0
        backward = 0
        for _ in range(epoch):
            start = time.time()
            # fprop through loop algo
            out = onloop(input)
            forward += time.time() - start

            start = time.time()
            # bprop through loop algo
            out.sum().backward()
            backward += time.time() - start

        self.logger.info(f'Loop Normalization Speed Test: '
                         f'Forward {forward * 1e6/1e5:.3f} us | '
                         f'Backward {backward * 1e6/1e5:.3f} us | '
                         f'Total {(forward + backward) * 1e6/1e5:.3f} us')

        self.logger.info('Make input tensors representative of size you will '
                         'use and then use the correct algorithm based on '
                         'speed of execution.')


if __name__ == '__main__':
    unittest.main()
