import torch
import torch.nn as nn
import torch.nn.functional as F
from models import simam_module

def xavier_init(layer):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
            
class ParametersExtracters_1(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        output_dim = cfg.num_filter_parameters
        # net = net - 0.5
        min_feature_map_size = 4
        print('extract_parameters CNN:')
        channels = cfg.base_channels
        
        self.conv_layers = nn.Sequential(
            # ex_conv0: 输入通道 3，输出通道 base_channels，kernel=3，下采样（stride=2）
            self._conv_block(3, channels, downsample=True),
            # ex_conv1: 输入通道 base_channels，输出 2*base_channels
            self._conv_block(channels, 2*channels, downsample=True),
            # ex_conv2: 输入输出通道均为 2*base_channels
            self._conv_block(2*channels, 2*channels, downsample=True),
            # ex_conv3
            self._conv_block(2*channels, 2*channels, downsample=True),
            # ex_conv4
            self._conv_block(2*channels, 2*channels, downsample=True),
        )
        
        # 全连接层
        self.fc1 = nn.Linear(4096, cfg.fc1_size)
        self.fc2 = nn.Linear(cfg.fc1_size, output_dim)
        
        self.apply(lambda m: xavier_init(m))
        
    def _conv_block(self,in_c, out_c, downsample=True):
        layers = [
            nn.Conv2d(in_c, out_c, 3, stride=2 if downsample else 1, padding=1, bias=True)
        ]
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.permute(0,3,1,2)
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        
        features = F.leaky_relu(self.fc1(x), 0.2)
        filter_features = self.fc2(features)
        
        return filter_features
    
class ParametersExtracters_2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        output_dim = cfg.num_filter_parameters
        # net = net - 0.5
        min_feature_map_size = 4
        # print('extract_parameters CNN:')
        channels = 16
        self.dropout_rate = cfg.dropout_rate
        
        if cfg.add_simam_attention:
            print("use SimAm attention module")
            Simam = simam_module.simam_module(cfg.e_lambda)
            self.conv_layers = nn.Sequential(
                # ex_conv0: 输入通道 3，输出通道 base_channels，kernel=3，下采样（stride=2）
                self._conv_block(3, channels, downsample=True),
                
                # ex_conv1: 输入通道 base_channels，输出 2*base_channels
                nn.Sequential(
                    self._conv_block(channels, 2*channels, downsample=True),
                    Simam
                ),
                
                # ex_conv2: 输入输出通道均为 2*base_channels
                nn.Sequential(
                    self._conv_block(2*channels, 2*channels, downsample=True),
                    Simam
                ),
                
                # ex_conv3
                nn.Sequential(
                    self._conv_block(2*channels, 2*channels, downsample=True),
                    Simam
                ),
                # ex_conv4
                self._conv_block(2*channels, 2*channels, downsample=True),
            )
        else:
            self.conv_layers = nn.Sequential(
                # ex_conv0: 输入通道 3，输出通道 base_channels，kernel=3，下采样（stride=2）
                self._conv_block(3, channels, downsample=True),
                # ex_conv1: 输入通道 base_channels，输出 2*base_channels
                self._conv_block(channels, 2*channels, downsample=True),
                # ex_conv2: 输入输出通道均为 2*base_channels
                self._conv_block(2*channels, 2*channels, downsample=True),
                # ex_conv3
                self._conv_block(2*channels, 2*channels, downsample=True),
                # ex_conv4
                self._conv_block(2*channels, 2*channels, downsample=True),
            )
        
        # 全连接层
        self.fc1 = nn.Linear(2048, 64)
        self.fc2 = nn.Linear(64, output_dim)
        
        self.apply(lambda m: xavier_init(m))
        
    def _conv_block(self, in_channels, out_channels, downsample=False):
        stride = 2 if downsample else 1
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,bias=True),
            nn.BatchNorm2d(out_channels),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=self.dropout_rate/2)  
        )
    
    def forward(self, x):
        # x = x.permute(0,3,1,2)
        x = self.conv_layers(x)
        # print(f"x type {x.dtype}")
        x = x.view(x.size(0), -1)
        # print("fc1 输入统计 - 均值:", x.mean(), " 最大值:", x.max(), " NaN:", torch.isnan(x).any())
    
        x = self.fc1(x)
        # print("fc1 输出统计 - 均值:", x.mean(), " 最大值:", x.max(), " NaN:", torch.isnan(x).any())
        
        features = F.leaky_relu(x, 0.2)
        # print("激活后统计 - 均值:", features.mean(), " 最大值:", features.max(), " NaN:", torch.isnan(features).any())
        # features = F.leaky_relu(self.fc1(x), 0.2)
        filter_features = self.fc2(features)
        if torch.isnan(filter_features).any():
            print("NaN occured in extractparam ")
        return filter_features
        
        
if __name__ == '__main__':
    import config
    x = torch.randn(1,3,256,256)
    model = ParametersExtracters_2(config.cfg)
    output = model(x)
    print(output)
        