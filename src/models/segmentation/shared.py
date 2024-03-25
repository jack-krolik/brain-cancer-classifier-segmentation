from torch import nn

class BatchNormRelu(nn.Module):
    def __init__(self, in_channels):
        super(BatchNormRelu, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)
    
    def _init_weights(self):
        """
        Initialize the weights of the model. 
        """
        nn.init.ones_(self.bnorm.weight)
        nn.init.zeros_(self.bnorm.bias)