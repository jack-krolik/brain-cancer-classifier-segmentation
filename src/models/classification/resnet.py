from torch import nn

class MultiResNetClassifier(nn.Module):
    def __init__(self, num_classes:int):
        super(MultiResNetClassifier, self).__init__()
        # Load ResNet model
        self.model = ResNet([2, 2, 2, 2], num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

class BinaryResNetClassifier(nn.Module):
    def __init__(self):
        super(BinaryResNetClassifier, self).__init__()
        # Load ResNet model
        self.model = ResNet([2, 2, 2, 2], num_classes=1)

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    """
    A residual block module for ResNet.
       
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride value for the convolutional layers. Default is 1.
        downsample (nn.Module, optional): Downsample module to be applied to the input. Default is None.

    Attributes:
        conv1 (nn.Sequential): First convolutional layer sequence.
        conv2 (nn.Sequential): Second convolutional layer sequence.
        downsample (nn.Module): Downsample module to be applied to the input.
        relu (nn.ReLU): ReLU activation function.
        out_channels (int): Number of output channels.

    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        # Define the first convolutional layer sequence
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # Define the second convolutional layer sequence
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        # Set the downsample module
        self.downsample = downsample

        # Define the ReLU activation function
        self.relu = nn.ReLU()

        # Set the number of output channels
        self.out_channels = out_channels

    def forward(self, x):
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        # Store the input tensor as the residual
        residual = x

        # Apply the first convolutional layer sequence
        out = self.conv1(x)

        # Apply the second convolutional layer sequence
        out = self.conv2(out)

        # Apply the downsample module if it exists
        if self.downsample:
            residual = self.downsample(x)

        # Add the residual to the output and apply the ReLU activation function
        out += residual
        out = self.relu(out)

        return out
    

class ResNet(nn.Module):
    """
    ResNet class for image classification.

    Args:
        layers (list): List of integers representing the number of residual blocks in each layer.
        num_classes (int, optional): Number of output classes. Defaults to 10.
    """

    def __init__(self, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 7x7 convolutional layer with stride 2 and padding 3
            nn.BatchNorm2d(64),  # Batch normalization layer for 64 channels
            nn.ReLU()  # ReLU activation function
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Max pooling layer with kernel size 3, stride 2, and padding 1
        self.layer0 = self._make_layer(64, layers[0], stride=1)  # First residual layer
        self.layer1 = self._make_layer(128, layers[1], stride=2)  # Second residual layer
        self.layer2 = self._make_layer(256, layers[2], stride=2)  # Third residual layer
        self.layer3 = self._make_layer(512, layers[3], stride=2)  # Fourth residual layer
        self.avgpool = nn.AvgPool2d(7, stride=1)  # Average pooling layer with kernel size 7 and stride 1
        self.fc = nn.Linear(512, num_classes)  # Fully connected layer with 512 input features and num_classes output features

    def _make_layer(self, planes, blocks, stride=1):
        """
        Helper method to create a residual layer.

        Args:
            block (nn.Module): The residual block module.
            planes (int): Number of output channels.
            blocks (int): Number of residual blocks in the layer.
            stride (int, optional): Stride value. Defaults to 1.

        Returns:
            nn.Sequential: Sequential module containing the residual blocks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),  # 1x1 convolutional layer with stride
                nn.BatchNorm2d(planes)  # Batch normalization layer for planes channels
            )
        layers = []
        layers.append(ResidualBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(ResidualBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ResNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv1(x)  # Apply the first convolutional layer sequence
        x = self.maxpool(x)  # Apply max pooling
        x = self.layer0(x)  # Apply the first residual layer
        x = self.layer1(x)  # Apply the second residual layer
        x = self.layer2(x)  # Apply the third residual layer
        x = self.layer3(x)  # Apply the fourth residual layer

        x = self.avgpool(x)  # Apply average pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)  # Apply the fully connected layer

        return x