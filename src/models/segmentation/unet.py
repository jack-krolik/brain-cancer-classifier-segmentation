# Implementation of the U-Net model inspired by the original paper: https://arxiv.org/pdf/1505.04597.pdf
# The model is implemented using the PyTorch framework.

import torch
from torch import nn
from functools import wraps

# TODO: move this to utils
def validate_input_shape(expected_shape: tuple):
    """
    Decorator to validate the input shape of a tensor.

    Args:
        expected_shape (tuple): The expected shape of the input tensor.
    """
    
    def decorator(func):
        @wraps(func)
        def wrapper(self, x, *args, **kwargs):
            if not isinstance(x, torch.Tensor):
                raise TypeError("Input must be a torch.Tensor")
            if x.shape[1:] != expected_shape:  # Assuming x.shape[0] is the batch size
                raise ValueError(f"Input tensor must have shape {expected_shape}, but got {x.shape[1:]}")
            return func(self, x, *args, **kwargs)
        return wrapper
    return decorator

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder Block
        self.conv_block_1 = UNetConvBlock(in_channels=3, out_channels1=64, out_channels2=64)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv_block_2 = UNetConvBlock(in_channels=64, out_channels1=128, out_channels2=128) 
        self.conv_block_3 = UNetConvBlock(in_channels=128, out_channels1=256, out_channels2=256)
        self.conv_block_4 = UNetConvBlock(in_channels=256, out_channels1=512, out_channels2=512)
        self.conv_block_5 = UNetConvBlock(in_channels=512, out_channels1=1024, out_channels2=1024)

        # Decoder Block
        self.upconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2) 
        self.conv_block_1 = UNetConvBlock(in_channels=1024, out_channels1=512, out_channels2=512) 
        self.upconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv_block_2 = UNetConvBlock(in_channels=512, out_channels1=256, out_channels2=256)
        self.upconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv_block_3 = UNetConvBlock(in_channels=256, out_channels1=128, out_channels2=128)
        self.upconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv_block_4 = UNetConvBlock(in_channels=128, out_channels1=64, out_channels2=64)

        # Output Layer
        # TODO - double check the output layer
        self.output = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, padding='valid') # TODO: double check on the padding
    
    # ensure x dimensions are Nx3x256x256
    @validate_input_shape((3, 256, 256)) # ignore the batch size
    def forward(self, x):
        """
        Forward pass through the UNet model.

        Note: Original paper uses 570x570 images, but this model will most likely use 256x256 images.
        """

        enc_1 = self.conv_block_1(x) # Input shape is 256x256x3 -> 252x252x64
        down_sample_1 = self.max_pool(dec_1) # Input shape is 252x252x64 -> 126x126x64
        enc_2 = self.conv_block_2(down_sample_1) # Input shape is 126x126x64 -> 122x122x128
        down_sample_2 = self.max_pool(enc_2) # Input shape is 122x122x128 -> 61x61x128
        enc_3 = self.conv_block_3(down_sample_2) # Input shape is 61x61x128 -> 57x57x256
        down_sample_3 = self.max_pool(enc_3) # Input shape is 57x57x256 -> 28x28x256
        enc_4 = self.conv_block_4(down_sample_3) # Input shape is 28x28x256 -> 24x24x512
        down_sample_4 = self.max_pool(env4) # Input shape is 24x24x512 -> 12x12x512
        encoding = self.conv_block_5(x) # Input shape is 12x12x512 -> 8x8x1024

        # Decoder Block
        up_sample_1 = self.upconv(encoding) # Input shape is 8x8x1024 -> 16x16x512
        dec_1 = torch.cat([up_sample_1, enc_4], dim=1) # Input shape is 16x16x512 -> 16x16x1024


        raise NotImplementedError('UNet model not implemented yet.')
    

class UNetConvBlock(nn.Module):
    # this should run the input between two convolutions with a ReLU activation function in between them and after the second one
    # need to ask for the input channel for first and output channel for both convolutions 
    def __init__(self, in_channels, out_channels1, out_channels2, kernel_size=3):
        super(UNetConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels1, kernel_size=kernel_size, padding='valid') # TODO: double check on the padding
        self.conv2 = nn.Conv2d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=kernel_size, padding='valid')

    def forward(self, x):
        x = self.conv1(x)
        nn.ReLU(x, inplace=True)
        x = self.conv2(x)
        nn.ReLU(x, inplace=True)
        return x