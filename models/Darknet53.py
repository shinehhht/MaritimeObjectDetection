import torch
from torch import nn

def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet53(nn.Module):
    def __init__(self, block):
        super(Darknet53, self).__init__()

        self.conv1 = conv_batch(3, 32,stride=1)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 256, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=256, num_blocks=2)
        self.conv4 = conv_batch(256, 512, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv5 = conv_batch(512, 1024, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=1024, num_blocks=8)
        self.conv6 = conv_batch(1024, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)
        # self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out) # [1, 64, 320, 320]
        # print(out.shape)
        out = self.conv3(out)
        out1 = self.residual_block2(out) # [1, 256, 160, 160]
        # print(out1.shape)
        out = self.conv4(out1)
        out2 = self.residual_block3(out) # [1, 512, 80, 80]
        # print(out2.shape)
        out = self.conv5(out2)
        out3 = self.residual_block4(out) # [1, 1024, 40, 40]
        # print(out3.shape)
        out = self.conv6(out3)
        out = self.residual_block5(out) # [1, 1024, 20, 20]
        # print(out.shape)
        # print(out.shape)
        # out = self.global_avg_pool(out)
        # out = out.view(-1, 1024)
        # out = self.fc(out)

        return (out, out3, out2)

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


def darknet53():
    return Darknet53(DarkResidualBlock)

if __name__ == '__main__':
    x = torch.randn(1,3,640,640)
    out = darknet53()(x)
    #print(out.shape)