# Define the Discriminator for the Generator Module in #2088
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    # Adversary to discriminate the protected image and the original image
    def __init__(self, blocks, in_channels, ngf=64):
        super(Discriminator, self).__init__()

        self.blocks = blocks
        self.conv_init = nn.Conv2d(in_channels, ngf, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(ngf)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(ngf, 1)
        

    def forward(self, image):
        x = self.conv_init(image)
        x = self.bn(x)
        x = self.relu(x)

        # Convolution
        for i in range(self.blocks - 1):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)

        # Adaptive Pooling
        x = self.pool(x)
        x = x.squeeze(3).squeeze(2) 
        x = self.linear(x)
        return x