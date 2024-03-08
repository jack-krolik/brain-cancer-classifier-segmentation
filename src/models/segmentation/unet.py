# Implementation of the U-Net model inspired by the original paper: https://arxiv.org/pdf/1505.04597.pdf
# The model is implemented using the PyTorch framework.

import torch
from torch import nn
from src.validators import validate_input_shape


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
    
    @validate_input_shape({
        'shape': (3, 320, 320),
        'dims': 4,
        'dtype': torch.float32,
        'device': None # ignore the device for now
    })
    def forward(self, x):
        """
        Forward pass through the UNet model.

        Note: Original paper uses 570x570 images, but this model will most likely use 256x256 images.
        320x320 is chosen as it won't cause rounding after maxpooling and is simply half of 640x640 (the dataset's original size).
        """

        # Encoder Block
        enc_1 = self.conv_block_1(x) # Input shape is 320x320x3 -> 320x320x64 
        down_sample_1 = self.max_pool(dec_1) # Input shape is 320x320x64 -> 160x160x64
        enc_2 = self.conv_block_2(down_sample_1) # Input shape is 160x160x64 -> 160x160x64
        down_sample_2 = self.max_pool(enc_2) # Input shape is 160x160x64 -> 80x80x128
        enc_3 = self.conv_block_3(down_sample_2) # Input shape is 80x80x128 -> 80x80x256
        down_sample_3 = self.max_pool(enc_3) # Input shape is 80x80x256 -> 40x40x256
        enc_4 = self.conv_block_4(down_sample_3) # Input shape is 40x40x256 -> 40x40x512
        down_sample_4 = self.max_pool(env4) # Input shape is 40x40x512 -> 20x20x512
        encoding = self.conv_block_5(x) # Input shape is 20x20x512 -> 20x20x1024

        # Decoder Block
        up_sample_1 = self.upconv(encoding) # Input shape is 20x20x1024 -> 40x40x512
        skip_1 = self._skip_connection(up_sample_1, enc_4) # Concatenate (40x40x512, 40x40x512) -> 40x40x1024
        dec_1 = self.conv_block_1(skip_1) # Input shape is 40x40x1024 -> 40x40x512
        up_sample_2 = self.upconv(dec_1)  # Input shape is 40x40x512 -> 80x80x256
        skip_2 = self._skip_connection(up_sample_2, enc_3)  # Concatenate (80x80x256, 80x80x256) -> 80x80x512
        dec_2 = self.conv_block_2(skip_2)  # Input shape is 80x80x512 -> 80x80x256
        up_sample_3 = self.upconv(dec_2) # Input shape is 80x80x256 -> 160x160x128
        skip_3 = self._skip_connection(up_sample_3, enc_2) # Concatenate (160x160x128, 160x160x128) -> 160x160x256
        dec_3 = self.conv_block_3(skip_3) # Input shape is 160x160x256 -> 160x160x128
        up_sample_4 = self.upconv(dec_3) # Input shape is 160x160x128 -> 320x320x64
        skip_4 = self._skip_connection(up_sample_4, enc_1) # Concatenate (320x320x64, 320x320x64) -> 320x320x128
        dec_4 = self.conv_block_4(skip_4)  # Input shape is 320x320x128 -> 320x320x64

        # Output Layer
        output = self.output(dec_4) # Input shape is 320x320x64 -> 320x320x2 TODO: determine if this is the correct output shape
        # specify, should the output be 320x320x2 or 320x320x1

        return output
    
    def _skip_connection(self, encode_component, decode_component):
        """
        Skip connection to concatenate the output of the encoder block with the input of the decoder block.

        Args:
            encode_component (torch.Tensor): The output of the encoder block. Shape is (N, C, H, W).
            decode_component (torch.Tensor): The input of the decoder block. Shape is (N, C, H, W).

        Where:
            N = batch size
            C = number of channels
            H = height
            W = width
        
        Note:
            The encoder and decoder W and H dimensions should be the same. But if they are not, then we
            need to resize the encoder output to match the decoder input.
        """

        # step 1: crop encoder output to match decoder input
        # get H and W dimensions of the encoder and decoder components
        decoder_shape = decode_component.shape[2:]
        encoder_shape = encode_component.shape[2:]
        if decoder_shape != encoder_shape:
            # crop the encoder output to match the decoder input H and W shape
            crop = (encoder_shape[0] - decoder_shape[0]) // 2
            encode_component = encode_component[:, :, crop:crop+decoder_shape[0], crop:crop+decoder_shape[1]]
        
        # step 2: concatenate the encoder and decoder components along the channel dimension
        return torch.cat((encode_component, decode_component), dim=1)
    

class UNetConvBlock(nn.Module):
    # this should run the input between two convolutions with a ReLU activation function in between them and after the second one
    # need to ask for the input channel for first and output channel for both convolutions 
    def __init__(self, in_channels, out_channels1, out_channels2, kernel_size=3):
        super(UNetConvBlock, self).__init__()

        # NOTE: padding is set to 'same' to ensure the output shape is the same as the input shape (this is different than the original paper, which uses 'valid' padding)
        # However, the original paper also uses strict 570x570 images, but this model will most likely use 268x268 images. 
        # padding='same' will ensure cleaner skip connections and output shapes. 
        # NOTE: padding='same' may also introduce artifacts at the edges of the images, this should be investigated further.
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels1, kernel_size=kernel_size, padding='same')# TODO: double check on the padding 
        self.conv2 = nn.Conv2d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=kernel_size, padding='same')

    def forward(self, x):
        x = self.conv1(x)
        nn.ReLU(x, inplace=True)
        x = self.conv2(x)
        nn.ReLU(x, inplace=True)
        return x