import torch
import torch.nn as nn
from torchvision import transforms
from models import CNNPP,yolo_fuse, Darknet53
import numpy as np
from PIL import Image
from pathlib import Path


base_dir = Path(__file__).parent.parent

class Combination(nn.Module):
    def __init__(self, cfg, weights, ch=3, nc=None, anchors=None,name = None):
        super().__init__()
        self.Yolo_net = yolo_fuse.Model(weights,ch, nc, anchors)
        self.Extract_param = CNNPP.ParametersExtracters_2(cfg)
        self.target_feature = Darknet53.darknet53()
        self.filters = cfg.filters 
        self.cfg = cfg
        filters = [x(cfg) for x in self.filters]
        self.filters = nn.ModuleList(filters)
        self.name = name
        
        
    def forward(self, input_data, origin_data, isp_flag,defog_A, IcA, batch_id=None, epoch=None, augment=False, profile=False):
        filtered_image_batch = input_data
        self.filter_params = input_data
        filter_imgs_series = []
        # print(f"input data type is {input_data[0].dtype}")
        if isp_flag:
            with torch.amp.autocast(device_type='cuda',dtype=torch.float32):
                resize_transform = transforms.Resize((256, 256))
                input_data = resize_transform(input_data)
                filter_features = self.Extract_param(input_data)
                #print(f"filter features is on {filter_features.device}")
                #print(f"image is on {filtered_image_batch.device}")
                #print(f"defog A is on {defog_A.device}")
                #print(f"ICA is on {IcA.device}")
                
                filter_parameters = []
                for j, filter in enumerate(self.filters):
                    # print(f"processed by {filter} filter")
                    filtered_image_batch, filter_parameter = filter.apply(filtered_image_batch, filter_features, defog_A, IcA) 
                    filter_parameters.append(filter_parameter)
                    filter_imgs_series.append(filtered_image_batch) # 每通过一个filter，记录一次image形式
            
            self.filter_params = filter_parameters   
        if not isp_flag:
            print(f"pure image type is {filtered_image_batch[0].dtype}")
            
        self.image_isped = filtered_image_batch
        self.filter_imgs_series = filter_imgs_series
        input_data = filtered_image_batch # 经过处理后的input data
        if torch.isnan(input_data).any():
            raise ValueError("Input images contain NaN values.")
        
        # print(f"data shape {input_data.shape}")
        
        if self.training:
            feature_origin = self.target_feature(origin_data) 
            pred = self.Yolo_net(input_data, feature_origin, augment, profile)
        else:
            print("now is eval")
            pred = self.Yolo_net(input_data, None, augment,profile)
        """

        feature_origin = self.target_feature(origin_data) # (b, 1024,20,20) / (b,1024,12,21)
        # print(f"feature shape {feature_origin.shape}")
        pred = self.Yolo_net(input_data, feature_origin, augment, profile)
        """
        
        test_data = input_data[0]
         
        if self.cfg.add_simam_attention:
            image_modified_path = str(base_dir)+'/data/modified/SimAm'
        else:
            image_modified_path = str(base_dir)+'/data/modified/'+ str(self.name) 
            
        if epoch and batch_id <= 20 and (epoch <= 10 or ((epoch+1) % 25 == 0)) :
            save_representitive_image(image_modified_path, test_data,batch_id,epoch)
        
        return pred
        
def save_representitive_image(path, image, batch_id, epoch):
    epoch_dir = Path(path+f"/epoch{epoch}")
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    image = image.detach().cpu().numpy()
    image_modified = np.transpose(image,(1,2,0))
    image_modified = (image_modified * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_modified)
    pil_image.save(str(epoch_dir)+f"/{batch_id}.jpg")
    
        