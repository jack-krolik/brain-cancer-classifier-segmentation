from torch import nn
from torchinfo import summary

from src.models.segmentation.unet import UNet

def main():
    # Create a UNet model
    model = UNet()
    # summary(model, input_size=(1, 3, 320, 320))
    summary(model)


if __name__ == '__main__':
    main()
