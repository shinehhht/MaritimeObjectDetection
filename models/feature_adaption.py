import torch
import torch.nn as nn
import torch.nn.functional as F
from models.odconv import ODConv2d
from models import simam_module

class ODConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=nn.BatchNorm2d,
                 reduction=0.0625, kernel_num=1):
        padding = (kernel_size - 1) // 2
        super(ODConvBNReLU, self).__init__(
            ODConv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups,
                     reduction=reduction, kernel_num=kernel_num),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )
        
        
class Adpation(nn.Module):
    def __init__(self,channel):
        super(Adpation,self).__init__()
        self.channel = channel
        self.odconv1 = ODConvBNReLU(self.channel,2*self.channel,kernel_size=1)
        self.odconv2 = ODConvBNReLU(2*self.channel,self.channel,kernel_size=1)
        self.simam = simam_module.simam_module(1e-4)
    def forward(self, features):
        adapted_list = []
        for feature in features:
            x = self.odconv1(feature)
            x = self.odconv2(x)
            x = self.simam(x)
            adapted_list.append(x)
        return adapted_list