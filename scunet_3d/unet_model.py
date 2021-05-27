import torch.nn as nn
from unet_parts import *
import torch
from config import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        x = x_channel
        self.inc = inconv(n_channels, x)
        
        self.down11 = down(x, x*2)
        self.down21 = down(x*2, x*4)
        self.down31 = down(x*4, x*8)
        self.down41 = down(x*8, x*16)

        self.up11 = up(x*16, x*8)
        self.up21 = up(x*8, x*4)
        self.up31 = up(x*4, x*2)
        self.up41 = up(x*2, x)
        
        self.unet_1st_out = outconv(x, n_channels)
        self.inc0 = inconv(n_channels*2, x)
        
        self.down12 = down(x, x*2)
        self.down22 = down(x*2, x*4)
        self.down32 = down(x*4, x*8)
        self.down42 = down(x*8, x*16)
        
        self.up12 = up(x*16, x*8)
        self.up22 = up(x*8, x*4)
        self.up32 = up(x*4, x*2)
        self.up42 = up(x*2, x)
        if is_3d and not in_out_same_size:
            self.unet_1st_out = outconv(x, n_channels)
        
        if in_out_same_size:
            self.unet_2nd_out = outconv(x, n_classes)
        else:            
            self.up52 = up_no_skipconn(x, x//2)
            self.unet_2nd_out = outconv(x//2, n_classes)

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
