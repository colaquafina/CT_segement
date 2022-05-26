import torch
from torch import nn
class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.down_conv1 = convolve(1, 64, 64)
        self.down_conv2 = convolve(64, 128, 128)
        self.down_conv3 = convolve(128, 256, 256)
        self.down_conv4 = convolve(256, 512, 512)
        self.down_conv5 = convolve(512, 1024, 1024)
        self.up4 = up(1024)
        self.up_conv4 = convolve(1024, 512, 512)
        self.up3 = up(512)
        self.up_conv3 = convolve(512, 256, 256)
        self.up2 = up(256)
        self.up_conv2 = convolve(256, 128, 128)
        self.up1 = up(128)
        self.up_conv1 = convolve(128, 64, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.out = nn.Conv2d(32, 1, kernel_size=3, padding="same")
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x1 = self.down_conv1(x)
        x2 = self.down_conv2(self.pool(x1))
        x3 = self.down_conv3(self.pool(x2))
        x4 = self.down_conv4(self.pool(x3))
        x = self.down_conv5(self.pool(x4))
        x = self.up4(x, x4)
        x = self.up_conv4(x)
        x = self.up3(x, x3)
        x = self.up_conv3(x)
        x = self.up2(x, x2)
        x = self.up_conv2(x)
        x = self.up1(x, x1)
        x = self.up_conv1(x)
        x = self.up0(x)
        x = self.out(x)
        return self.sigmoid(x)
        






class convolve(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, kernel_size = 3):
        super().__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels = in_channel, out_channels = mid_channel, kernel_size = kernel_size, padding = "same"),
        nn.BatchNorm2d(mid_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels = mid_channel, out_channels = out_channel, kernel_size = kernel_size, padding = "same"),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layer(x)


class up(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.layer = nn.ConvTranspose2d(in_channel, in_channel // 2, 2, 2)
    
    def forward(self, x1, x2):
        x1 = self.layer(x1)
        diff_h = x2.size()[2] - x1.size()[2]
        diff_w = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, (diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2))
        return torch.cat([x1, x2], dim=1)

