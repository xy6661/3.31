from torch import nn
import torch

from .cga import SpatialAttention, ChannelAttention, PixelAttention


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn

        pattn2 = self.pa(initial, pattn1)

        result = pattn2 * x + (1 - pattn2) * y        #方案三
        result = self.conv(result)
        return result


