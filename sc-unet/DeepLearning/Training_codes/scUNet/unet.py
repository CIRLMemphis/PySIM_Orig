# full assembly of the sub-parts to form the complete net
import torch.nn as nn
from unet_parts import *
import torch

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc1 = inconv(n_channels, 64)
        self.down11 = down(64, 128)
        self.down21 = down(128, 256)
        self.up11 = up(256, 128)
        self.up21 = up(128, 64)
        self.unet_1st_out = outconv(64, n_channels)

        self.inc2 = inconv(n_channels*2, 64)
        self.down12 = down(64, 128)
        self.down22 = down(128, 256)
        self.up12 = up(256, 128)
        self.up22 = up(128, 64)
        self.unet_2nd_out = outconv(64, n_channels)


        self.inc3 = inconv(n_channels*2, 64)
        self.down13 = down(64, 128)
        self.down23 = down(128, 256)
        self.up13 = up(256, 128)
        self.up23 = up(128, 64)
        self.unet_3rd_out = outconv(64, n_channels)
 
        self.inc4 = inconv(n_channels*2, 64)
        self.down14 = down(64, 128)
        self.down24 = down(128, 256)
        self.up14 = up(256, 128)
        self.up24 = up(128, 64)
        self.unet_4th_out = outconv(64, n_channels)

        self.inc5 = inconv(n_channels*2, 64)
        self.down15 = down(64, 128)
        self.down25 = down(128, 256)
        self.up15 = up(256, 128)
        self.up25 = up(128, 64)
        self.up35 = up_no_skipconn(64, 32)
        self.unet_5th_out = outconv(32, n_classes)

        

    def forward(self, x):
        x_in = x
        x1 = self.inc1(x)
        x2 = self.down11(x1)
        x3 = self.down21(x2)
        x = self.up11(x3, x2)
        x = self.up21(x2, x1)
        x = self.unet_1st_out(x)
        x = torch.cat([x_in, x], dim=1)

        x1 = self.inc2(x)
        x2 = self.down12(x1)
        x3 = self.down22(x2)
        x = self.up12(x3, x2)
        x = self.up22(x2, x1)
        x = self.unet_2nd_out(x)
        x = torch.cat([x_in, x], dim=1)

        x1 = self.inc3(x)
        x2 = self.down13(x1)
        x3 = self.down23(x2)
        x = self.up13(x3, x2)
        x = self.up23(x2, x1)
        x = self.unet_3rd_out(x)
        x = torch.cat([x_in, x], dim=1)

        x1 = self.inc4(x)
        x2 = self.down14(x1)
        x3 = self.down24(x2)
        x = self.up14(x3, x2)
        x = self.up24(x2, x1)
        x = self.unet_4th_out(x)
        x = torch.cat([x_in, x], dim=1)

        x1 = self.inc5(x)
        x2 = self.down15(x1)
        x3 = self.down25(x2)
        x = self.up15(x3, x2)
        x = self.up25(x2, x1)
        x = self.up35(x, x1)
        x = self.unet_5th_out(x)
        return x
