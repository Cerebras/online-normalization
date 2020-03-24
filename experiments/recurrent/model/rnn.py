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
Implementation of RNN in Pytorch

Most of the object / method structure, and code modified from
[Pytorch's RNN](https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html).
Note this is heavily modified.
"""


class RNNCell(nn.RNNCellBase):
    r"""An Elman RNN cell with tanh or ReLU non-linearity.

    RNNCell with normalization. This class adds layer or online normalization
    to an RNNCell.

    .. math::

        h' = \tanh(w_{ih} x + b_{ih}  +  w_{hh} h + b_{hh})

    If :attr:`nonlinearity` is `'relu'`, then ReLU is used in place of tanh.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'

    Inputs: input, hidden
        - **input** of shape `(batch, input_size)`: tensor containing input features
        - **hidden** of shape `(batch, hidden_size)`: tensor containing the initial hidden
          state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: h'
        - **h'** of shape `(batch, hidden_size)`: tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_[ih]h: the learnable [input/hidden]-hidden weights, of shape
                      `(hidden_size x (input_size + hidden_size))`
        bias_[ih]h: the learnable bias, of shape `(hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    Examples::

        >>> rnn = nn.RNNCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
                hx = rnn(input[i], hx)
                output.append(hx)
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh",
                 norm=None, **kwargs):
        super(RNNCell, self).__init__(input_size=input_size,
                                      hidden_size=hidden_size,
                                      bias=bias, num_chunks=1)
        self.nonlinearity = nonlinearity

        if nonlinearity == "tanh":
            self.nonlin = torch.tanh
        elif nonlinearity == "relu":
            self.nonlin = torch.relu
        elif nonlinearity == "none":
            warnings.warn('RNN not using a non-linearity')
            self.nonlin = None
        else:
            raise RuntimeError("Unknown nonlinearity: {}".format(self.nonlinearity))

        if norm is None:
            warnings.warn('RNNCell w/out norm see Pytorch\'s RNNCell: '
                          'https://pytorch.org/docs/stable/nn.html')

        self.norm = None
        if norm == 'layer_norm':
            warnings.warn('RNN Using LayerNorm')
            self.norm = nn.LayerNorm(hidden_size)
            self.reset_norm_parameters()
        elif norm == 'online_norm':
            warnings.warn('RNN Using OnlineNorm')
            self.norm = OnlineNorm1D(
                hidden_size, batch_size=kwargs['batch_size'],
                alpha_fwd=kwargs['alpha_fwd'], alpha_bkw=kwargs['alpha_bkw'], 
                ecm=kwargs['ecm']
            )
            self.reset_norm_parameters()
        elif norm == 'none':
            warnings.warn('RNN Not Using A Norm')
            self.norm = None
        elif norm is not None:
            raise ValueError(f"Unknown nonlinearity norm {norm}")
        else:
            warnings.warn('RNN Not Using A Norm')

    def reset_norm_parameters(self):
        if self.norm:
            nn.init.constant_(self.norm.weight, 1)
            nn.init.zeros_(self.norm.bias)

    def forward(self, input, hidden):
        self.check_forward_input(input)
        if hidden is None:
            hidden = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        self.check_forward_hidden(input, hidden)

        # Linear mappings
        h_1 = F.linear(torch.cat((input, hidden), dim=1),
                       torch.cat((self.weight_ih, self.weight_hh), dim=1),
                       bias=self.bias_ih + self.bias_hh)

        # apply norm
        if self.norm is not None:
            h_1 = self.norm(h_1)

        # apply non-linearity
        if self.nonlin is not None:
            h_1 = self.nonlin(h_1)

        return h_1


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False,
                 norm=None, nonlinearity="tanh", **kwargs):
        super(RNN, self).__init__()
        assert not bidirectional, 'functionality not implemented'
        self.num_directions = 2 if bidirectional else 1
        self.batch_first = batch_first
        self.bias = bias
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        if norm is not None:
            if norm[0].lower() == 'o':
                norm = 'online_norm'
            elif norm[0].lower() == 'l':
                norm = 'layer_norm'
            elif norm[0].lower() == 'n':
                norm = None
            else:
                raise ValueError(f"Unknown nonlinearity norm {norm}")

        # cache first layer
        self.rnn_cells = [RNNCell(input_size=input_size,
                                  hidden_size=hidden_size,
                                  bias=bias, norm=norm,
                                  nonlinearity=nonlinearity, **kwargs)]

        # cache all subsequent layers
        for cell_idx in range(1, num_layers):
            self.rnn_cells += [RNNCell(input_size=hidden_size,
                                       hidden_size=hidden_size,
                                       bias=bias, norm=norm,
                                       nonlinearity=nonlinearity, **kwargs)]

        self.set_rnn_modules()

    def set_rnn_modules(self):
        for i, rnn_cell in enumerate(self.rnn_cells):
            setattr(self, 'rnn_layer_' + str(i), rnn_cell)

    def forward(self, input, hidden):
        out = []
        for input_t in input:
            hidden[0] = self.rnn_cells[0](input_t, hidden[0])
            if self.num_layers - 1:
                hidden[0] = F.dropout(hidden[0], p=self.dropout,
                                      training=self.training)
            for cell_idx in range(1, self.num_layers):
                hidden[cell_idx] = self.rnn_cells[cell_idx](hidden[cell_idx - 1],
                                                            hidden[cell_idx])
                # add dropout to all but last layer
                if cell_idx != self.num_layers - 1:
                    hidden[cell_idx] = F.dropout(hidden[cell_idx],
                                                 p=self.dropout,
                                                 training=self.training)
            out += [hidden[-1]]
        out = torch.stack(out)
        return out, hidden
