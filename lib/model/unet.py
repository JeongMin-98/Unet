# --------------------------------------------------------
# Reference from milesial/Pytorch-UNet
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
# ----------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """ 3x3 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ConvBlock(nn.Module):
    """ Convolution Block : conv3x3 => BN => ReLU """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            # DoubleConv(in_channels, out_channels)
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # # if bilinear, use the normal convolutions to reduce the number of channels
        # if bilinear:
        #     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # else:
        # my Unet doesn't use bilinear. The number of out_channels is half of the number of in_channels.
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv_1x1(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Contracting Path
        self.input_features = DoubleConv(n_channels, 64)
        out_channel = 64
        self.contracting_path = []
        for i in range(1, 5):
            in_channel = out_channel
            out_channel = out_channel * 2
            self.contracting_path.append(Down(in_channel, out_channel))
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # self.down4 = Down(512, 1024)

        # Expansive Path
        self.expansive_path = []
        for i in range(1, 5):
            in_channel = out_channel
            out_channel = out_channel // 2
            self.expansive_path.append(Up(in_channel, out_channel))
        # self.up1 = Up(1024, 512)
        # self.up2 = Up(512, 256)
        # self.up3 = Up(256, 128)
        # self.up4 = Up(128, 64)
        self.output = DoubleConv(128, 64)
        self.segmentor = Conv1x1(64, n_classes)

    def forward(self, x):
        ipt = self.input_features.forward(x)
        compress_layer_output = [ipt]
        for i in range(4):
            compress_layer_output.append(self.contracting_path[i].forward(compress_layer_output[-1]))

        expansive_layer_output = compress_layer_output[-1]
        for i in range(3):
            expansive_layer_output = self.expansive_path[i].forward(expansive_layer_output,
                                                                    compress_layer_output[3 - i])
        output_feature = self.output.forward(expansive_layer_output)
        logits = self.segmentor.forward(output_feature)
        return logits
