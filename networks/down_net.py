import torch
import torch.nn as nn
import torch.nn.functional as F
from .deform import DeformConv2d

def default_conv(in_channels, out_channels, kernel_size):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=True)


class Dehazeblock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(Dehazeblock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size)
        self.conv3 = DeformConv2d(dim,dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res+x
        res = self.conv2(res)
        res = self.conv3(res)
        res += x
        return res

# Adaptive Feature Fusion Module
class AFFM(nn.Module):
    def __init__(self, m=-0.80,channel=None):
        super(AFFM, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)

        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))

        return out

class DehazeNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3):
        super(DehazeNet, self).__init__()

        # Downsampling
        self.down1 = nn.Sequential( nn.ReflectionPad2d(3),
                                    nn.Conv2d(input_nc, 64, 7),
                                    nn.ReLU(inplace=True))

        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                   nn.ReLU(inplace=True) )

        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                   nn.ReLU(inplace=True) )

        # DFP-Net block
        self.Dehazeblock= Dehazeblock(conv=default_conv,dim=256,kernel_size=3)

        # Upsampling
        self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(inplace=True))

        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(inplace=True))

        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(64, output_nc, 7)
                                 )

        self.AFFM1 = AFFM(m=-1,channel=256)
        self.AFFM2 = AFFM(m=-0.6,channel=128)

    def forward(self, x):

        x_down1 = self.down1(x)           # [bs, 64, 256, 256]
        x_down2 = self.down2(x_down1)     # [bs, 128, 128, 128]
        x_down3 = self.down3(x_down2)     # [bs, 256, 64, 64]

        x1 = self.Dehazeblock(x_down3)    # [bs, 256, 64, 64]
        x2 = self.Dehazeblock(x1)         # [bs, 256, 64, 64]
        x3 = self.Dehazeblock(x2)         # [bs, 256, 64, 64]
        x4 = self.Dehazeblock(x3)         # [bs, 256, 64, 64]
        x5 = self.Dehazeblock(x4)         # [bs, 256, 64, 64]
        x6 = self.Dehazeblock(x5)         # [bs, 256, 64, 64]

        x_out_affm = self.AFFM1(x_down3, x6)
        x_up1 = self.up1(x_out_affm)      # [bs, 128, 128, 128]
        x_up1_affm = self.AFFM2(x_down2, x_up1)
        x_up2 = self.up2(x_up1_affm)      # [bs, 64, 256, 256]
        out = self.up3(x_up2)             # [bs,  3, 256, 256]

        return out







if __name__ =="__main__":
    x=torch.ones(1,3,256,256)
    D=DehazeNet()
    res=D(x)
    print(res.shape)