import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#Guided images filtering for grayscale images
class GuidedFilter(nn.Module):

    def __init__(self, r, eps, gpu_ids=None):  # only work for gpu case at this moment
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = nn.AvgPool2d(kernel_size=2 * self.r + 1, stride=1, padding=self.r)

    def forward(self, I, p):
        """
        I -- guidance image, should be [0, 1]
        p -- filtering input image, should be [0, 1]
        """
        N = self.boxfilter(torch.ones(p.size()))

        if I.is_cuda:
            N = N.cuda()

        mean_I = self.boxfilter(I) / N
        mean_p = self.boxfilter(p) / N
        mean_Ip = self.boxfilter(I * p) / N
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.boxfilter(I * I) / N
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I
        mean_a = self.boxfilter(a) / N
        mean_b = self.boxfilter(b) / N

        return mean_a * I + mean_b


class GLGenerator(nn.Module):
    """Create a global local generator"""

    def __init__(self, r=21, eps=0.01):
        super(GLGenerator, self).__init__()

        self.guided_filter = GuidedFilter(r=r, eps=eps)

    def forward(self, x):

        n, c, h, w = x.size()
        # get GuidedFilter initial results
        res_global = self.guided_filter(x, x)

        # 将x和y在C维度上拆分为c个单通道张量
        x_list = torch.chunk(x, c, dim=1)
        y_list = torch.chunk(res_global, c, dim=1)

        # 对拆分后的单通道张量在C维度上逐一进行torch.sub()操作
        z_list = []
        for i in range(c):
            z_list.append(torch.sub(x_list[i], y_list[i]))

        # 将拆分后的单通道张量在C维度上拼接
        res_local = torch.cat(z_list, dim=1)

        return res_global, res_local




