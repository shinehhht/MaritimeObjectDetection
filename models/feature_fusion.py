import torch
import torch.nn as nn
import torch.nn.functional as F

class FFM(nn.Module):
    def __init__(self, r, channels=1024):
        super(FFM, self).__init__()
        
        self.activation = nn.Sigmoid()
        self.size = r
        self.conv_com = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def forward(self, processed_feature, origin_feature):
        # print(f"processed type {processed_feature.shape}")
        # print(f"origin type {origin_feature.shape}")
        assert processed_feature.shape == origin_feature.shape
        pointwise_addition = processed_feature + origin_feature
        residual = pointwise_addition
        if (processed_feature.shape[-1] % 2 != 0 or processed_feature.shape[-2] % 2 != 0):
            x = F.avg_pool2d(pointwise_addition, kernel_size=self.size)
            x = self.conv_com(x)
            x = F.interpolate(x, size=(processed_feature.shape[-2], processed_feature.shape[-1]), mode='bilinear', align_corners=False) 
        else:
            x = F.avg_pool2d(pointwise_addition, kernel_size=self.size)
            x = self.conv_com(x)
            x = self.upsample(x)
        # print(f"x shape {x.shape}")
        # weight = self.activation(x + residual)
        weight = self.activation(x)
        # print(weight.shape)
        processed = self.conv(processed_feature)
        origin = self.conv(origin_feature)
        # print(f"origin shape {origin.shape}")
        out = weight*processed + (1-weight)*origin
        # print(out.shape)
        return out
          
if __name__ == '__main__':
    origin = torch.randn(8,1024,12,21)
    proessed = torch.randn(8,1024,12,21)
    
    # origin = torch.randn(8,1024,20,20)
    # proessed = torch.randn(8,1024,20,20)
    ffm = FFM(2,1024)
    x = ffm(proessed,origin)
    #print(x.shape)
        