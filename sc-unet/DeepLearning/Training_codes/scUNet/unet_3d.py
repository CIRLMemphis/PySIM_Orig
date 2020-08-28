# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from torchsummary import summary

class double_conv3d(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        #print('----------------',in_ch,out_ch)
        super(double_conv3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv3d, self).__init__()
        self.conv = double_conv3d(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down3d, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1,2,2)),
            double_conv3d(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x



class up_no_skipconn3d(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up_no_skipconn3d, self).__init__()
        self.up_no_skipconn = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(1,2,2),stride=(1,2,2))

    def forward(self, x1, x2):
        x2 = self.up_no_skipconn(x1)
        return x2

class up3d(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up3d, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_ch, out_ch,kernel_size=(1,2,2),stride=(1,2,2))

        self.conv = double_conv3d(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x1


class outconv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv3d, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x




class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3D, self).__init__()
        self.inc = inconv3d(n_channels, 64)
        self.down11 = down3d(64, 128)
        self.down21 = down3d(128, 256)
        self.down31 = down3d(256, 512)
        self.down41 = down3d(512, 1024)
        self.up11 = up3d(1024, 512)
        self.up21 = up3d(512, 256)
        self.up31 = up3d(256, 128)
        self.up41 = up3d(128, 64)
        self.unet_1st_out = outconv3d(64, n_channels)
        
        self.inc0 = inconv3d(n_channels*2, 64)
        self.down12 = down3d(64, 128)
        self.down22 = down3d(128, 256)
        self.down32 = down3d(256, 512)
        self.down42 = down3d(512, 1024)
        self.up12 = up3d(1024, 512)
        self.up22 = up3d(512, 256)
        self.up32 = up3d(256, 128)
        self.up42 = up3d(128, 64)
        if in_out_same_size:
            self.unet_2nd_out = outconv3d(64, n_classes)
        else:
            self.up52 = up_no_skipconn3d(64, 32)
            self.unet_2nd_out = outconv3d(32, n_classes)


        

    def forward(self, x):
        x_in = x
        x1 = self.inc(x)
        x2 = self.down11(x1)
        x3 = self.down21(x2)
        x4 = self.down31(x3)
        x5 = self.down41(x4)
        x = self.up11(x5, x4)
        x = self.up21(x4, x3)
        x = self.up31(x, x2)
        x = self.up41(x, x1)
        x = self.unet_1st_out(x)
        x = torch.cat([x_in, x], dim=1)
        x1 = self.inc0(x)
        x2 = self.down12(x1)
        x3 = self.down22(x2)
        x4 = self.down32(x3)
        x5 = self.down42(x4)
        x = self.up12(x5, x4)
        x = self.up22(x4, x3)
        x = self.up32(x, x2)
        x = self.up42(x, x1)
        if not in_out_same_size:
            x = self.up52(x, x1)
        x = self.unet_2nd_out(x)
        return x


class UNetTest(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetTest, self).__init__()
        self.inc = inconv3d(n_channels, 64)
        self.down11 = down3d(64, 128)
        self.up11 = up3d(128, 64)
        #self.down21 = down3d(128, 256)
        #self.down31 = down3d(256, 512)
        #self.down41 = down3d(512, 1024)

    def forward(self, x):
        x_in = x
        x1 = self.inc(x)
        x2 = self.down11(x1)
        x = self.up11(x2, x1)
        #x3 = self.down21(x2)
        #x4 = self.down31(x3)
        #x5 = self.down41(x4)
        return x



cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet3D(n_channels=15, n_classes=3)
model.cuda(cuda)
#print(summary(model,(15,3,128,128)))