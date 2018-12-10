import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

# Unet model derived from https://github.com/milesial/Pytorch-UNet
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self,scale=0.5):
        super().__init__()
    def forward(self,x):
        return F.interpolate(x,scale_factor=0.5)

class interpDownConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.downConv = nn.Sequential(
            Downsample(),
            double_conv(in_ch,out_ch))
    def forward(self,x):
        x = self.downConv(x)
        return x

class UNetGenerator(nn.Module):
    def __init__(self,n_channels=6):
        super().__init__()
        self.inc = inconv(n_channels, 16)
        self.down1 = interpDownConv(16, 32)
        self.down2 = interpDownConv(32, 64)
        self.down3 = interpDownConv(64, 128)
        self.down4 = interpDownConv(128, 128)
        self.up1 = up(256, 64)
        self.up2 = up(128, 32)
        self.up3 = up(64, 16)
        self.up4 = up(32, 16)
        self.outc = double_conv(16,3)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
class UNetDiscriminator(nn.Module):
    def __init__(self, n_channels=3,height=512,width=288,hidden_size=300):
        super().__init__()
        self.inc = inconv(n_channels, 16)
        self.down1 = interpDownConv(16, 32)
        self.down2 = interpDownConv(32, 64)
        self.down3 = interpDownConv(64, 128)
        self.down4 = interpDownConv(128, 128)
        self.up1 = up(256, 64)
        self.up2 = up(128, 32)
        self.up3 = up(64, 16)
        self.up4 = up(32, 16)
        self.outc = double_conv(16, 3)
        self.outLinear = nn.Linear(height * width * 3, hidden_size)
        self.flatten = Flatten()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.flatten(x)
        x = self.outLinear(x)
        return F.sigmoid(x)