import torch
from torch import nn
from efficientnet_pytorch import EfficientNet


class Classifier(nn.Module):
    """
    Uses the outputs of the BiFPN to create predictions for each anchor.
    """

    def __init__(self, in_channels, hidden_layers, n_anchors, n_classes) -> None:
        super(Classifier, self).__init__()

        self.in_channels = in_channels
        self.conv_layers = hidden_layers
        self.n_anchors = n_anchors  # at each spacial location
        self.n_classes = n_classes

        self.classification_net = nn.Sequential()

        layers = []
        for _ in range(hidden_layers):
            layers += [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                ),
                nn.ReLU(),
            ]

        # adding last layers
        layers += [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.n_anchors * self.n_classes,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            # nn.Sigmoid(),
        ]

        self.class_net = nn.Sequential(*layers)

    def forward(self, x):

        # output shape: N x (n_classes * n_anchors) x W x H
        output = self.class_net(x)

        # dimension extraction
        batch_size, channels, w, h = output.shape

        # -> shape: : N x W x H (n_classes * n_anchors_at_each_loc)
        output = output.permute(0, 2, 3, 1)

        # -> shape: N x W x H x n_anchors_at_each_loc x n_classes
        output = output.contiguous().view(
            batch_size, w, h, self.n_anchors, self.n_classes
        )

        # -> shape: N x (all_anchors) x n_classes
        output = output.contiguous().view(batch_size, -1, self.n_classes)

        return output


class Regressor(nn.Module):
    """
    Uses the outputs of the BiFPN to create anchor offset predictions.
    """

    def __init__(self, in_channels, n_anchors, hidden_layers) -> None:
        super(Regressor, self).__init__()

        self.in_channels = in_channels
        self.n_anchors = n_anchors  # at each spacial location
        self.hidden_layers = hidden_layers
        layers = []
        for _ in range(hidden_layers):
            layers += [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                ),
                nn.ReLU(),
            ]

        layers += [
            nn.Conv2d(in_channels, n_anchors * 4, kernel_size=3, stride=1, padding=1)
        ]

        self.reg_net = nn.Sequential(*layers)

    def forward(self, x):

        # shape: [N x (n_anchors * 4), w, h]
        output = self.reg_net(x)

        # dimension extraction
        batch_size, channels, w, h = output.shape

        # -> shape: : N x W x H (n_anchors_per_location * 4)
        output = output.permute(0, 2, 3, 1)

        # -> shape: : N x (all_anchors) x 4
        output = output.contiguous().view(batch_size, -1, 4)

        return output


class ConvBlock(nn.Module):
    """
    Refernace: https://github.com/signatrix/efficientdet/blob/master/src/model.py
    """

    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                num_channels,
                num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=num_channels,
            ),
            nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=num_channels, momentum=0.9997, eps=4e-5),
            nn.ReLU(),
        )

    def forward(self, input):
        return self.conv(input)


class BiFPBlock(nn.Module):
    """
    Referance: https://github.com/signatrix/efficientdet/blob/master/src/model.py
    """

    def __init__(self, num_channels, epsilon=1e-4):
        super(BiFPBlock, self).__init__()
        self.epsilon = epsilon
        # Conv layers
        self.conv6_up = ConvBlock(num_channels)
        self.conv5_up = ConvBlock(num_channels)
        self.conv4_up = ConvBlock(num_channels)
        self.conv3_up = ConvBlock(num_channels)
        self.conv4_down = ConvBlock(num_channels)
        self.conv5_down = ConvBlock(num_channels)
        self.conv6_down = ConvBlock(num_channels)
        self.conv7_down = ConvBlock(num_channels)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p5_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p4_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p3_upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.p4_downsample = nn.MaxPool2d(kernel_size=2)
        self.p5_downsample = nn.MaxPool2d(kernel_size=2)
        self.p6_downsample = nn.MaxPool2d(kernel_size=2)
        self.p7_downsample = nn.MaxPool2d(kernel_size=2)

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2))
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2))
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2))
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2))
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3))
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3))
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3))
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2))
        self.p7_w2_relu = nn.ReLU()

    def forward(self, inputs):
        """
        P7_0 -------------------------- P7_2 -------->

        P6_0 ---------- P6_1 ---------- P6_2 -------->

        P5_0 ---------- P5_1 ---------- P5_2 -------->

        P4_0 ---------- P4_1 ---------- P4_2 -------->

        P3_0 -------------------------- P3_2 -------->
        """

        # P3_0, P4_0, P5_0, P6_0 and P7_0
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs
        # P7_0 to P7_2
        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in))
        # Weights for P5_0 and P6_0 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_0 to P5_1 respectively
        p5_up = self.conv5_up(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up))
        # Weights for P4_0 and P5_0 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up))

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            weight[0] * p4_in
            + weight[1] * p4_up
            + weight[2] * self.p4_downsample(p3_out)
        )
        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            weight[0] * p5_in
            + weight[1] * p5_up
            + weight[2] * self.p5_downsample(p4_out)
        )
        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            weight[0] * p6_in
            + weight[1] * p6_up
            + weight[2] * self.p6_downsample(p5_out)
        )
        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(
            weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)
        )

        return p3_out, p4_out, p5_out, p6_out, p7_out


class EfficientDet(nn.Module):
    def __init__(
        self,
        pretrained_backbone,
        n_classes,
        n_anchors,
        bifpn_layers,
        n_channels=64,
    ) -> None:
        """
        Model used to perform all stages of object detection.
        """

        super(EfficientDet, self).__init__()
        self.n_classes = n_classes

        self.n_channels = n_channels
        self.bifpn_layers = bifpn_layers
        self.n_anchors = n_anchors
        self.fpn_sizes = []

        # EfficientNet Backbone
        if pretrained_backbone:
            efficientnet = EfficientNet.from_pretrained("efficientnet-b0")
        else:
            efficientnet = EfficientNet.from_name("efficientnet-b0")

        # There are 4 blocks from EfficientNet-B0
        blocks = []
        for block in efficientnet._blocks:
            blocks.append(block)
            if block._depthwise_conv.stride == [2, 2]:
                self.fpn_sizes.append(block._project_conv.out_channels)

        self.backbone = nn.Sequential(
            efficientnet._conv_stem, efficientnet._bn0, *blocks
        )

        # Conv operations between backbone and BiFPN
        self.conv3 = nn.Conv2d(
            self.fpn_sizes[1], self.n_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv4 = nn.Conv2d(
            self.fpn_sizes[2], self.n_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv5 = nn.Conv2d(
            self.fpn_sizes[3], self.n_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv6 = nn.Conv2d(
            self.fpn_sizes[3], self.n_channels, kernel_size=3, stride=2, padding=1
        )
        self.conv7 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                self.n_channels, self.n_channels, kernel_size=3, stride=2, padding=1
            ),
        )

        # Bi-directional Pyramid Feature Network
        self.bifpn = nn.Sequential(
            *[BiFPBlock(self.n_channels) for _ in range(bifpn_layers)]
        )

        # Regressor
        self.regressor = Regressor(
            in_channels=self.n_channels, n_anchors=self.n_anchors, hidden_layers=1
        )

        # Classifier
        self.classifier = Classifier(
            in_channels=self.n_channels,
            n_anchors=self.n_anchors,
            n_classes=n_classes,
            hidden_layers=3,
        )

        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Number of parameter: {self.total_params}")

    def forward(self, x):

        # Forward pass of the EfficientNet-B0 backbone
        features = []
        x = self.backbone[0](x)
        x = self.backbone[1](x)
        for block in self.backbone[2:]:
            x = block(x)
            if block._depthwise_conv.stride == [2, 2]:
                features.append(x)

        # conv layer between backbone and BiFPN
        p3 = self.conv3(features[1])
        p4 = self.conv4(features[2])
        p5 = self.conv5(features[3])
        p6 = self.conv6(features[3])
        p7 = self.conv7(p6)

        # Forward pass of the BiFPN
        features = self.bifpn([p3, p4, p5, p6, p7])

        # Classification of each anchor for each class
        # only using the highest-res features to make predictions
        class_preds = self.classifier(features[0])

        # Adjustment of each anchor for each class
        # only using the highest-res features to make predictions
        anchor_adjustments = self.regressor(features[0])

        return class_preds, anchor_adjustments


def save_checkpoint(model, file_path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved to {file_path}")


def load_checkpoint(model, file_path):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint loaded from {file_path}")
