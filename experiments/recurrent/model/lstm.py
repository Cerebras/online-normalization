"""
Released under BSD 3-Clause License,
Modifications are Copyright (c) 2019 Cerebras, Inc.
All rights reserved.
"""
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from online_norm_pytorch import OnlineNorm1D, ControlNorm1DLoop

"""
Implementation of LSTM in pytorch

Most of the object / method structure, and code modified from
[Pytorch's LSTM](https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html).
Note this is heavily modified.
"""

class LSTMCellCustom(nn.RNNCellBase):

    r"""
    A reimplementation of Hochreiter & Schmidhuber 'Long-Short Term Memory' for
    pytorch:
    http://www.bioinf.jku.at/publications/older/2604.pdf

    Signature should match pytorch's LSTMCell:
    https://pytorch.org/docs/stable/nn.html#lstmcell

    This implementation has the same representational power as pytorch's
    LSTMCell but uses `(4*hidden_size)` less parameters, saves the computation
    needed to add the second bias and the matrix multiply is done with one
    kernel call instead of 2.


    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

    Inputs: input, (h_0, c_0)
        - **input** of shape `(batch, input_size)`: tensor containing input
          features
        - **h_0** of shape `(batch, hidden_size)`: tensor containing the
          initial hidden state for each element in the batch.
        - **c_0** of shape `(batch, hidden_size)`: tensor containing the
          initial cell state for each element in the batch.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to
          zero.

    Outputs: h_1, c_1
        - **h_1** of shape `(batch, hidden_size)`: tensor containing the next
          hidden state for each element in the batch
        - **c_1** of shape `(batch, hidden_size)`: tensor containing the next
          cell state for each element in the batch

    Attributes:
        weight_[ih]h: the learnable [input/hidden]-hidden weights, of shape
                      `(4 * hidden_size x (input_size + hidden_size))`
                      NOTE: 
                        1. In Pytorch's LSTMCell this is separated into
                           weight_ih and weight_hh
                        2. In Pytorch's LSTMCell W is split into
                           (W_i|W_f|W_g|W_o),
                           here this is split into (W_i|W_f|W_o|W_g)
        bias_[ih]h: the learnable bias, of shape `(4*hidden_size)`
                    NOTE: 
                        1. In Pytorch's LSTMCell this is separated into bias_ih
                           and bias_hh
                        2. In Pytorch's LSTMCell W is split into
                           (b_i|b_f|b_g|b_o),
                           here this is split into (b_i|b_f|b_o|b_g)


    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    """

    def __init__(self, input_size, hidden_size, bias=True, **kwargs):
        super(LSTMCellCustom, self).__init__(input_size=input_size,
                                             hidden_size=hidden_size,
                                             bias=bias, num_chunks=4)

    def forward(self, input, hx):
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size,
                                 requires_grad=False)
            hx = (hx.clone(), hx.clone())
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')

        h_0, c_0 = hx

        # Linear mappings
        lstm_gates = F.linear(torch.cat((input, h_0), dim=1),
                              torch.cat((self.weight_ih, self.weight_hh), dim=1),
                              bias=self.bias_ih + self.bias_hh)

        # activations
        gates = lstm_gates[:, :3 * self.hidden_size].sigmoid()
        i_gate, f_gate, o_gate = gates.chunk(3, -1)
        g_gate = lstm_gates[:, 3 * self.hidden_size:].tanh()

        # recurrent state computations
        c_1 = c_0 * f_gate + i_gate * g_gate
        h_1 = o_gate * c_1.tanh()

        return h_1, c_1


class NormLSTMCell(nn.RNNCellBase):
    r"""
    LSTMCell with normalization. This class adds layer normalization to
    a LSTMCell.

    LSTMCell implementation based on Hochreiter & Schmidhuber:
    'Long-Short Term Memory'

        http://www.bioinf.jku.at/publications/older/2604.pdf

    Signature should match pytorch's LSTMCell:

        https://pytorch.org/docs/stable/nn.html#lstmcell


    normalization is applied before the 5 internal nonlinearities unless
    `cell_norm` is set to false

    If norm[0].lower() == 'l' layer normalization is selected
    Layer normalization implementation is based on:

        https://arxiv.org/abs/1607.06450.

    "Layer Normalization"
    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

    If norm[0].lower() == 'o' online normalization is selected
    Online normalization implementation is based on:

        https://papers.nips.cc/paper/9051-online-normalization-for-training-neural-networks

    "Online Normalization for Training Neural Networks"
    Vitaliy Chiley, Ilya Sharapov, Atli Kosson, Urs Koster, Ryan Reece,
    Sofia Samaniego de la Fuente, Vishal Subbiah, Michael James
    """

    def __init__(self, input_size, hidden_size, bias=True,
                 norm=None, cell_norm=True, **kwargs):
        super(NormLSTMCell, self).__init__(input_size=input_size,
                                           hidden_size=hidden_size,
                                           bias=bias, num_chunks=4)
        self.reset_parameters()

        self.cell_norm = norm and cell_norm
        num_norms = 5 if self.cell_norm else 4

        self.norms = None
        if not norm:
            warnings.warn('LSTMCell w/out LayerNorm see Pytorch\'s LSTMCell: '
                          'https://pytorch.org/docs/stable/nn.html#lstmcell')
        if norm[0].lower() == 'l':
            warnings.warn('Using Layer Norm in LSTMCell')
            self.norms = [nn.LayerNorm(hidden_size) for _ in range(num_norms)]
        elif norm[0].lower() == 'o':
            warnings.warn('Using Online Norm in LSTMCell')
            self.norms = [
                OnlineNorm1D(hidden_size, batch_size=kwargs['batch_size'],
                             alpha_fwd=kwargs['alpha_fwd'],
                             alpha_bkw=kwargs['alpha_bkw'], 
                             ecm=kwargs['ecm']) for _ in range(num_norms)]

        self.reset_norm_parameters()
        self.set_norm_modules()

    def reset_norm_parameters(self):
        if self.norms:
            for norm in self.norms:
                nn.init.constant_(norm.weight, 1)
                nn.init.zeros_(norm.bias)

    def set_norm_modules(self):
        if self.norms:
            for i, norm in enumerate(self.norms):
                setattr(self, 'norm_layer_' + str(i), norm)

    def forward(self, input, hx):
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size,
                                 requires_grad=False)
            hx = (hx.clone(), hx.clone())
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')

        h_0, c_0 = hx

        # Linear mappings
        lstm_gates = F.linear(torch.cat((input, h_0), dim=1),
                              torch.cat((self.weight_ih, self.weight_hh), dim=1),
                              bias=self.bias_ih + self.bias_hh).chunk(4, -1)

        # apply norms
        if self.norms is not None:
            gates = []
            for g, norm in zip(lstm_gates, self.norms):
                gates += [norm(g)]
        else:
            gates = lstm_gates

        # activations
        i_gate = gates[0].sigmoid()
        f_gate = gates[1].sigmoid()
        o_gate = gates[2].sigmoid()
        g_gate = gates[3].tanh()

        # recurrent state computations
        c_1 = c_0 * f_gate + i_gate * g_gate
        if self.cell_norm and self.norms is not None:
            warnings.warn('Normalizing Cell Gate of LSTM')
            c_1 = self.norms[-1](c_1)
        h_1 = o_gate * c_1.tanh()

        return h_1, c_1


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False,
                 norm=None, cell_norm=True, **kwargs):
        super(LSTM, self).__init__()
        assert not bidirectional, 'functionality not implemented'
        self.num_directions = 2 if bidirectional else 1
        self.batch_first = batch_first
        self.bias = bias
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # cache first layer
        if norm:
            self.lstm_cells = [NormLSTMCell(input_size=input_size,
                                            hidden_size=hidden_size,
                                            bias=bias, norm=norm,
                                            cell_norm=cell_norm, **kwargs)]
        else:
            self.lstm_cells = [LSTMCellCustom(input_size=input_size,
                                              hidden_size=hidden_size,
                                              bias=bias, **kwargs)]

        # cache all subsequent layers
        for _ in range(1, num_layers):
            if norm:
                self.lstm_cells += [NormLSTMCell(input_size=hidden_size,
                                                 hidden_size=hidden_size,
                                                 bias=bias, norm=norm,
                                                 cell_norm=cell_norm,
                                                 **kwargs)]
            else:
                self.lstm_cells += [LSTMCellCustom(input_size=hidden_size,
                                                   hidden_size=hidden_size,
                                                   bias=bias, **kwargs)]

        self.set_lstm_modules()

    def set_lstm_modules(self):
        for i, lstm_cell in enumerate(self.lstm_cells):
            setattr(self, 'lstm_layer_' + str(i), lstm_cell)

    def forward(self, input, hx):
        out = []
        if hx is not None:
            h, c = hx[0], hx[1]
        for input_t in input:
            h[0], c[0] = self.lstm_cells[0](input_t, (h[0], c[0]))
            if self.num_layers - 1:
                h[0] = F.dropout(h[0], p=self.dropout, training=self.training)
            for lstm_idx in range(1, self.num_layers): 
                h[lstm_idx], c[lstm_idx] = self.lstm_cells[lstm_idx](h[lstm_idx - 1], (h[lstm_idx], c[lstm_idx]))
                # add dropout to all but last layer
                if lstm_idx != self.num_layers - 1:
                    h[lstm_idx] = F.dropout(h[lstm_idx], p=self.dropout,
                                            training=self.training)
            out += [h[-1]]
        out = torch.stack(out)
        return out, (h, c)
