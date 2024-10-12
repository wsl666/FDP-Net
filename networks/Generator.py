import torch
import torch.nn as nn
from .PreGenerator import GLGenerator
from .up_net import UNet
from .down_net import DehazeNet as down_net


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.Preprocess = GLGenerator()
        self.up_net = UNet()
        self.down_net = down_net()

    def forward(self, x):

        x_global, x_local = self.Preprocess(x)

        res_local = self.up_net(x_local)

        res_global = self.down_net(x_global)

        out = res_global + res_local

        out = torch.clamp(out, -1,1)

        return out


if __name__=="__main__":
    x=torch.randn(1,3,256,256).cuda()
    net=Generator().cuda()
    a=net(x)
    print("a的shape",a.shape)


