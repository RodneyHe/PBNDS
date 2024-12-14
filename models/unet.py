import torch
from torch import nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False),
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
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

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
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = (x - x.min()) / (x.max() - x.min())
        return x

class Encoder(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        
        self.inchns = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 512))
        self.down5 = (Down(512, 512))
        #self.down6 = (Down(512, 512))
    
    def forward(self, x):
        
        x1 = self.inchns(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        #x7 = self.down6(x6)
        
        return [x6, x5, x4, x3, x2, x1]

class Decoder(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        self.up1 = (Up(1024, 512))
        #self.up2 = (Up(1024, 512))
        self.up3 = (Up(1024, 256))
        self.up4 = (Up(512, 128))
        self.up5 = (Up(256, 64))
        self.up6 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))
    
    def forward(self, feature_list):
        
        feature_iter = iter(feature_list)
        
        x = self.up1(next(feature_iter), next(feature_iter))
        #x = self.up2(x, next(feature_iter))
        x = self.up3(x, next(feature_iter))
        x = self.up4(x, next(feature_iter))
        x = self.up5(x, next(feature_iter))
        x = self.up6(x, next(feature_iter))
        logits = self.outc(x)
        
        return logits

class LightDecoder(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        self.up1 = (Up(1024, 512))
        self.up2 = (Up(1024, 512))
        self.up3 = (Up(1024, 256))
        self.up4 = (Up(512, 128))
        self.up5 = (Up(256, 64))
        self.outc = (OutConv(64, n_classes))

    def forward(self, feature_list):
        
        feature_iter = iter(feature_list)
        
        x = self.up1(next(feature_iter), next(feature_iter))
        x = self.up2(x, next(feature_iter))
        x = self.up3(x, next(feature_iter))
        x = self.up4(x, next(feature_iter))
        x = self.up5(x, next(feature_iter))
        logits = self.outc(x)
        
        return logits

class UNet128(nn.Module):
    def __init__(self, in_chns, out_chns):
        super().__init__()

        self.encoder = Encoder(in_chns)
        self.decoder = Decoder(out_chns)

    def forward(self, x):
        
        feature_list =self.encoder(x)
        pred_features = self.decoder(feature_list)
        
        return pred_features
