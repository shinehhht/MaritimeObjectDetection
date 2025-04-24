import torch
import torch.nn as nn
from models import yolo_adaption, Darknet53
from pathlib import Path


base_dir = Path(__file__).parent.parent

class Combination(nn.Module):
    def __init__(self, weights, ch=3, nc=None, anchors=None,name = None):
        super().__init__()
        self.Yolo_net = yolo_adaption.Model(weights,ch, nc, anchors)
        self.extract_clear_feature = Darknet53.darknet53()
        self.name = name
        
    def forward(self, input_data, origin_data, augment=False, profile=False):
        if self.training:
            origin_feature = self.extract_clear_feature(origin_data) # 3 scales ( (b,c,h,w), (b,c,h,w), (b,c,h,w) )   
            print(f"origin feature length {len(origin_feature)}, shape {origin_feature[0].shape}")
            clear_features, adapted_features, pred = self.Yolo_net(input_data, origin_feature, augment, profile)
            return clear_features, adapted_features, pred 
        else:
            print("now is eval")
            _, _, pred = self.Yolo_net(input_data, None, augment,profile)
            return None, None, pred
        
    
        