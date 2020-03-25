"""
Released under BSD 3-Clause License,
Modifications are Copyright (c) 2019 Cerebras, Inc.
All rights reserved.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(in_channels, out_channels, kernel_size=3,
                norm_layer=None, norm_kwargs={}):
    return nn.Sequential(OrderedDict([
        ('conv0', nn.Conv2d(in_channels, out_channels, kernel_size, padding=1,
                  bias=(norm_kwargs['mode'] is None or norm_kwargs['mode'] is 'none'))),
        ('norm0', norm_layer(out_channels, **norm_kwargs)),
        ('reul0', nn.ReLU(inplace=True)),
        ('conv1', nn.Conv2d(out_channels, out_channels, kernel_size, padding=1,
                  bias=(norm_kwargs['mode'] is None or norm_kwargs['mode'] is 'none'))),
        ('norm1', norm_layer(out_channels, **norm_kwargs)),
        ('reul1', nn.ReLU(inplace=True))
        ])
    )   


class UNet(nn.Module):
    """
    Pytorch implementation of 
    [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
    by Ronneberger et al.
    """

    def __init__(self, n_class, norm_layer=None, norm_kwargs={}):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64,
                                       norm_layer=norm_layer,
                                       norm_kwargs=norm_kwargs)
        self.dconv_down2 = double_conv(64, 128,
                                       norm_layer=norm_layer,
                                       norm_kwargs=norm_kwargs)
        self.dconv_down3 = double_conv(128, 256,
                                       norm_layer=norm_layer,
                                       norm_kwargs=norm_kwargs)
        self.dconv_down4 = double_conv(256, 512,
                                       norm_layer=norm_layer,
                                       norm_kwargs=norm_kwargs)        

        self.maxpool = nn.MaxPool2d(2)
        
        self.dconv_up3 = double_conv(256 + 512, 256,
                                     norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs)
        self.dconv_up2 = double_conv(128 + 256, 128,
                                     norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs)
        self.dconv_up1 = double_conv(128 + 64, 64,
                                     norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = F.interpolate(x, scale_factor=2,
                          mode='bilinear', align_corners=True)
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = F.interpolate(x, scale_factor=2,
                          mode='bilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = F.interpolate(x, scale_factor=2,
                          mode='bilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
