"""
Released under BSD 3-Clause License,
Copyright (c) 2019 Cerebras Systems Inc.
All rights reserved.
"""
import torch.nn as nn

from .norm import LayerNorm1d
from online_norm_pytorch import OnlineNorm1D

__all__ = ['MLP_Model', 'mlp_model']


class MLP_Layer(nn.Module):

    def __init__(self, in_shape, out_shape, relu=True,
                 norm_layer=None, norm_kwargs={}):
        super(MLP_Layer, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape

        fc_bias = norm_layer is None or norm_layer is 'none'
        self.fc = nn.Linear(in_shape, out_shape, bias=fc_bias)
        self.norm = norm_layer(out_shape, **norm_kwargs)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        x = self.norm(x)
        return self.relu(x) if self.relu is not None else x


class MLP_Model(nn.Module):

    def __init__(self, in_shape=784, num_classes=10, depth=3,
                 depth_hidden=[500, 300], norm_layer=None, norm_kwargs={}):
        super(MLP_Model, self).__init__()
        self.classes = num_classes
        
        self.in_layer = MLP_Layer(in_shape, depth_hidden[0],
                                  norm_layer=norm_layer,
                                  norm_kwargs=norm_kwargs)
        self.hidden_layers = None
        if depth > 2:
            self.hidden_layers = nn.Sequential(*[MLP_Layer(depth_hidden[i],
                                                           depth_hidden[i + 1],
                                                           norm_layer=norm_layer,
                                                           norm_kwargs=norm_kwargs) for i in range(depth - 2)])
        self.out_layer = MLP_Layer(depth_hidden[-1], num_classes, relu=False,
                                   norm_layer=norm_layer,
                                   norm_kwargs={'mode': None})


        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, LayerNorm1d, OnlineNorm1D)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.in_layer(x)
        if self.hidden_layers is not None:
            x = self.hidden_layers(x)
        return self.out_layer(x)


def mlp_model(**kwargs):
    """ Constructs a MLP Model for fmnist training """
    return MLP_Model(**kwargs)
