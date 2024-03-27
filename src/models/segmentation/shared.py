from torch import nn

class BatchNormRelu(nn.Module):
    def __init__(self, in_channels, activation_variant="relu"):
        super(BatchNormRelu, self).__init__()
        activation = nn.ReLU()
        if activation_variant == "prelu":
            activation = nn.PReLU()

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            activation
        )

    def forward(self, x):
        return self.block(x)
    
    def _init_weights(self):
        """
        Initialize the weights of the model. 
        """
        nn.init.ones_(self.bnorm.weight)
        nn.init.zeros_(self.bnorm.bias)