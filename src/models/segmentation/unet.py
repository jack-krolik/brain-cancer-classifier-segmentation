# Implementation of the U-Net model inspired by the original paper: https://arxiv.org/pdf/1505.04597.pdf
# The model is implemented using the PyTorch framework.

import torch
from torch import nn
import torch.nn.functional as F
from src.utils.validators import validate_model_input


class UNet(nn.Module):
    # TODO: need to ask for input channels and output channels for the output layer
    # TODO: need to determine if intermediate layers should be affected by input in_channels and output out_channels
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder Block
        self.enc_conv_block_1 = UNetConvBlock(in_channels=3, out_channels1=64, out_channels2=64)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.enc_conv_block_2 = UNetConvBlock(in_channels=64, out_channels1=128, out_channels2=128) 
        self.enc_conv_block_3 = UNetConvBlock(in_channels=128, out_channels1=256, out_channels2=256)
        self.enc_conv_block_4 = UNetConvBlock(in_channels=256, out_channels1=512, out_channels2=512)
        self.enc_conv_block_5 = UNetConvBlock(in_channels=512, out_channels1=1024, out_channels2=1024)

        # Decoder Block
        self.upconv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2) 
        self.dec_conv_block_1 = UNetConvBlock(in_channels=1024, out_channels1=512, out_channels2=512) 
        self.upconv_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.dec_conv_block_2 = UNetConvBlock(in_channels=512, out_channels1=256, out_channels2=256)
        self.upconv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.dec_conv_block_3 = UNetConvBlock(in_channels=256, out_channels1=128, out_channels2=128)
        self.upconv_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.dec_conv_block_4 = UNetConvBlock(in_channels=128, out_channels1=64, out_channels2=64)

        # Output Layer
        # TODO - double check the output layer
        self.output = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, padding='valid') # TODO: double check on the padding

        self._init_weights()
    
    @validate_model_input({
        'shape': (3, 320, 320), #TODO: might need to check self.in_channels instead of assuming 3
        'dims': [3,4],
        'dtype': torch.float32,
        'device': None # ignore the device for now
    })
    def forward(self, x):
        """
        Forward pass through the UNet model.

        Note: Original paper uses 570x570 images, but this model will most likely use 320x320 images.
        320x320 is chosen as it won't cause rounding after maxpooling and is simply half of 640x640 (the dataset's original size).
        """

        # Encoder Block
        enc_1 = self.enc_conv_block_1(x) # Input shape is 3x320x320 -> 64x320x320
        down_sample_1 = self.max_pool(enc_1) # Input shape is 64x320x320 -> 64x160x160
        enc_2 = self.enc_conv_block_2(down_sample_1) # Input shape is 64x160x160 -> 128x160x160
        down_sample_2 = self.max_pool(enc_2) # Input shape is 128x160x160 -> 128x80x80
        enc_3 = self.enc_conv_block_3(down_sample_2) # Input shape is 128x80x80 -> 256x80x80
        down_sample_3 = self.max_pool(enc_3) # Input shape is 256x80x80 -> 256x40x40
        enc_4 = self.enc_conv_block_4(down_sample_3) # Input shape is 256x40x40 -> 512x40x40
        down_sample_4 = self.max_pool(enc_4) # Input shape is 512x40x40 -> 512x20x20
        encoding = self.enc_conv_block_5(down_sample_4) # Input shape is 512x20x20 -> 1024x20x20

        # Decoder Block
        up_sample_1 = self.upconv_1(encoding) # Input shape is 1024x20x20 -> 512x40x40
        skip_1 = self._skip_connection(up_sample_1, enc_4) # Concatenate (512x40x40, 512x40x40) -> 1024x40x40
        dec_1 = self.dec_conv_block_1(skip_1) # Input shape is 1024x40x40 -> 512x40x40
        up_sample_2 = self.upconv_2(dec_1)  # Input shape is 512x40x40 -> 256x80x80
        skip_2 = self._skip_connection(up_sample_2, enc_3)  # Concatenate (256x80x80, 256x80x80) -> 512x80x80
        dec_2 = self.dec_conv_block_2(skip_2)  # Input shape is 512x80x80 -> 256x80x80
        up_sample_3 = self.upconv_3(dec_2) # Input shape is 256x80x80 -> 128x160x160
        skip_3 = self._skip_connection(up_sample_3, enc_2) # Concatenate (128x160x160, 128x160x160) -> 256x160x160
        dec_3 = self.dec_conv_block_3(skip_3) # Input shape is 256x160x160 -> 128x160x160
        up_sample_4 = self.upconv_4(dec_3) # Input shape is 128x160x160 -> 64x320x320
        skip_4 = self._skip_connection(up_sample_4, enc_1) # Concatenate (64x320x320, 64x320x320) -> 128x320x320
        dec_4 = self.dec_conv_block_4(skip_4)  # Input shape is 128x320x320 -> 64x320x320

        # Output Layer
        output = self.output(dec_4) # Input shape is 64x320x320 -> 2x320x320
        # NOTE: output currently has 2 channels which implies a binary classification problem.

        # softmax activation function over the channel dimension
        channel_dim = 1 if len(output.shape) == 4 else 0
        softmaxed_output = F.softmax(output, dim=channel_dim)

        return softmaxed_output
    
    def _skip_connection(self, decode_component, encode_component):
        """
        Skip connection to concatenate the output of the encoder block with the input of the decoder block.

        Args:
            decode_component (torch.Tensor): The input of the decoder block. Shape is (N, C, H, W).
            encode_component (torch.Tensor): The output of the encoder block. Shape is (N, C, H, W).

        Where:
            N = batch size
            C = number of channels
            H = height
            W = width
        
        Note:
            The encoder and decoder W and H dimensions should be the same. But if they are not, then we
            need to resize the encoder output to match the decoder input.
        """

        # determine dimension of H and W
        height_dim = 2 if len(decode_component.shape) == 4 else 1

        # step 1: crop encoder output to match decoder input
        # get H and W dimensions of the encoder and decoder components
        decoder_shape = decode_component.shape[height_dim:]
        encoder_shape = encode_component.shape[height_dim:]
        if decoder_shape != encoder_shape:
            # crop the encoder output to match the decoder input H and W shape
            crop = (encoder_shape[0] - decoder_shape[0]) // 2
            encode_component = encode_component[:, :, crop:crop+decoder_shape[0], crop:crop+decoder_shape[1]]
        
        # step 2: concatenate the encoder and decoder components along the channel dimension
        cat_dim = 1 if len(decode_component.shape) == 4 else 0
        return torch.cat((encode_component, decode_component), dim=cat_dim)
    
    def _init_weights(self):
        """
        Initialize the weights of the model. 
        I follow the initialization scheme used in the original U-Net paper. 
        Moreover, each conv param ~ N(0, sqrt(2/(kernel_size*kernel_size*in_channels)))
        """
        def init_weights(m):
            if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                nn.init.normal_(m.weight, std=m.in_channels * m.kernel_size[0] * m.kernel_size[1])

        # all conv layers
        self.dec_conv_block_1._init_weights()
        self.dec_conv_block_2._init_weights()
        self.dec_conv_block_3._init_weights()
        self.dec_conv_block_4._init_weights()
        self.enc_conv_block_1._init_weights()
        self.enc_conv_block_2._init_weights()
        self.enc_conv_block_3._init_weights()
        self.enc_conv_block_4._init_weights()
        self.enc_conv_block_5._init_weights()

        # all upconv layers
        init_weights(self.upconv_1)
        init_weights(self.upconv_2)
        init_weights(self.upconv_3)
        init_weights(self.upconv_4)
    

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

        # TODO: might need to do a BatchNorm before the ReLU

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        self.relu(x)
        x = self.conv2(x)
        self.relu(x)
        return x
    
    def _init_weights(self):
        """
        Initialize the weights of the model. 
        I follow the initialization scheme used in the original U-Net paper. 
        Moreover, each conv param ~ N(0, sqrt(2/(kernel_size*kernel_size*in_channels)))
        """

        # all conv layers
        nn.init.normal_(self.conv1.weight, std=self.conv1.in_channels * self.conv1.kernel_size[0] * self.conv1.kernel_size[1])
        nn.init.normal_(self.conv2.weight, std=self.conv2.in_channels * self.conv2.kernel_size[0] * self.conv2.kernel_size[1])