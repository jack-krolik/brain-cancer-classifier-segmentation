# Implementation of LETNet model for segmentation inspired by the paper: https://arxiv.org/pdf/2302.10484.pdf

import torch
from torch import nn

from src.models.segmentation.shared import BatchNormRelu

class LETNet(nn.Module):
    def __init__(self):
        super(LETNet, self).__init__()
        self.init_block = nn.Sequential(
            ConvBNPRelu(3, 32, kernel_size=3, padding=1),
            ConvBNPRelu(32, 32, kernel_size=3, padding=1),
            ConvBNPRelu(32, 32, stride=2, kernel_size=3, padding=1),
        )

        # TODO: downsample from 32 to 64 channels
        self.downsample = nn.Sequential(
            ConvBNPRelu(32, 64, stride=2, kernel_size=3, padding=1),
        )
        
    
    def forward(self, x):
        raise NotImplementedError("LETNet forward pass not implemented yet")

class LBD(nn.Module):
    def __init__(self, in_channels: int, dilation: int = 1):
        super(LBD, self).__init__()
        
        half_in_channels = in_channels // 2

        self.in_block = nn.Sequential(
            ConvBNPRelu(in_channels, half_in_channels, kernel_size=1, padding=0),
            ConvBNPRelu(half_in_channels, half_in_channels, kernel_size=(3, 1), padding=(1, 0)),
            ConvBNPRelu(half_in_channels, half_in_channels, kernel_size=(1, 3), padding=(0, 1)),
        )

        # depth wise convolution sequence
        # NOTE: goal is to capture local information in the image
        self.local_split = nn.Sequential(
            ConvBNPRelu(half_in_channels, half_in_channels, kernel_size=(3, 1), padding=(1, 0), groups=half_in_channels),
            ConvBNPRelu(half_in_channels, half_in_channels, kernel_size=(1, 3), padding=(0, 1), groups=half_in_channels),
            # TODO: add Channel Attention Block
        )

        # atrous convolution sequence (dilated convolution)
        # NOTE: goal is to capture distant information in the image
        self.distant_split = nn.Sequential(
            ConvBNPRelu(half_in_channels, half_in_channels, kernel_size=(3, 1), padding=(dilation, 0), groups=half_in_channels),
            ConvBNPRelu(half_in_channels, half_in_channels, kernel_size=(1, 3), padding=(0, dilation), groups=half_in_channels),
            # TODO: add Channel Attention Block
        )

        self.final_conv = ConvBNPRelu(half_in_channels, in_channels, kernel_size=1, padding=0)

        # TODO: add shuffling operation
    
    def forward(self, x):
        x = self.in_block(x)
        local = self.local_split(x)
        distant = self.distant_split(x)
        x = local + distant
        x = self.final_conv(x)
        # TODO: add shuffling operation
        raise NotImplementedError("Shuffling operation + Channel Attention Block not implemented yet")

# normalized Convolutional Layer with Batch Normalization and ReLU activation
class ConvBNPRelu(nn.Module):
    """
    Convolutional Layer with Batch Normalization and PReLU activation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, stride: int = 1, groups: int = 1):
        super(ConvBnRelu, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups),
            BatchNormRelu(out_channels, activation_variant="prelu")
        )

    def forward(self, x):
        return self.block(x)

    def _init_weights(self):
        """
        Initialize the weights of the model. 
        """
        nn.init.ones_(self.bnorm.weight)
        nn.init.zeros_(self.bnorm.bias)
        # TODO: add additional initialization for the convolutional layer

        
        

