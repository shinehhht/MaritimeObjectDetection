import torch
import torch.nn as nn
import torch.nn.functional as F
from util_filters import lrelu, rgb2lum, tanh_range, lerp

class Filter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    # self.height, self.width, self.channels = list(map(int, net.get_shape()[1:]))

        # Specified in child classes
        self.num_filter_parameters = None
        self.short_name = None
        self.filter_parameters = None
        
    def get_short_name(self):
        assert self.short_name
        return self.short_name

    def get_num_filter_parameters(self):
        assert self.num_filter_parameters
        return self.num_filter_parameters

    def get_begin_filter_parameter(self):
        return self.begin_filter_parameter
    
    def extract_parameters(self, features):
        return features[:, self.get_begin_filter_parameter():(self.get_begin_filter_parameter() + self.get_num_filter_parameters())], \
           features[:, self.get_begin_filter_parameter():(self.get_begin_filter_parameter() + self.get_num_filter_parameters())]
           
    def filter_param_regressor(self, features): # 将feature值约束到指定范围（类似激活）
        assert False

    # Process the whole image, without masking
    # Should be implemented in child classes
    def process(self, img, param, defog, IcA):
        assert False

    def debug_info_batched(self):
        return False

    def no_high_res(self):
        return False
    
    def apply(self, img, img_features=None, defog_A=None, IcA=None, specified_parameter=None, high_res=None ):
        assert (img_features is None) ^ (specified_parameter is None)
        if img_features is not None:
            filter_features, mask_parameters = self.extract_parameters(img_features)
            filter_parameters = self.filter_param_regressor(filter_features)
            
        if high_res is not None:
        # working on high res...
            pass
        debug_info = {}
        # We only debug the first image of this batch
        if self.debug_info_batched():
            debug_info['filter_parameters'] = filter_parameters
        else:
            debug_info['filter_parameters'] = filter_parameters[0]
        # self.mask_parameters = mask_parameters
        # self.mask = self.get_mask(img, mask_parameters)
        # debug_info['mask'] = self.mask[0]
        #low_res_output = lerp(img, self.process(img, filter_parameters), self.mask)
        low_res_output = self.process(img, filter_parameters, defog_A, IcA)

        if high_res is not None:
            if self.no_high_res():
                high_res_output = high_res
        else:
            high_res_output = None
        #return low_res_output, high_res_output, debug_info
        return low_res_output, filter_parameters
  

        def draw_high_res_text(self, text, canvas):
            cv2.putText(
                canvas,
                text, (30, 128),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0),
                thickness=5)
            return canvas
        

class DefogFilter(Filter):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.short_name = 'DF'
        self.begin_filter_parameter = cfg.defog_begin_param
        self.num_filter_parameters = 1
    
    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.defog_range)(features)
    
    def process(self, img, param, defog_A, IcA):
        # print('      defog_A:', img.shape) # (B,C,H,W)
        # print('      defog_A:', IcA.shape) # (B,1,H,W)
        # print('      defog_A:', defog_A.shape)
        assert not torch.isnan(img).any(), "输入图像含NaN"
        assert not torch.isnan(param).any(), "参数param含NaN"
        assert not torch.isnan(IcA).any(), "IcA含NaN"
        assert not torch.isnan(defog_A).any(), "defog_A含NaN"
        # print("defog filter processing")
        
        tx = 1 - param[:, None, None, :]*IcA
        tx = torch.clamp(tx, min=0.01)
        """
        # tx = 1 - 0.5*IcA
        tx_1 = tx.repeat(1,3,1,1) # (B,C,H,W)
        tx_1 = torch.clamp(tx_1, min=0.01)

        defog_A = defog_A.view(-1, 3, 1, 1)
        # print(defog_A.shape)
        # print(tx_1.shape)
        """
        if torch.isnan(tx).any():
            print("NaN occured in tx ")
            return img  # 返回原始图像防止崩溃

        tx_1 = tx.repeat(1, 3, 1, 1)  # 扩展通道
        tx_1 = torch.clamp(tx_1, min=0.01)
        if torch.isnan(tx_1).any():
            print("NaN occured in tx_1 ")
            return img

        defog_A = defog_A.view(-1, 3, 1, 1)
        if torch.isnan(defog_A).any():
            print("NaN occured in defog_A ")
            return img
        # 4. 去雾公式计算
        result = (img - defog_A) / tx_1 + defog_A
        return result
      
class ImprovedWhiteBalanceFilter(Filter):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.short_name = 'W'
        self.channels = 3
        self.begin_filter_parameter = cfg.wb_begin_param
        self.num_filter_parameters = self.channels

    def filter_param_regressor(self, features):
        log_wb_range = 0.5
        mask = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32).to(features.device)
        # mask = np.array(((1, 0, 1)), dtype=np.float32).reshape(1, 3)
        # print(mask.shape)
        assert mask.shape == (1, 3)
        features = features * mask
        color_scaling = torch.exp(tanh_range(-log_wb_range, log_wb_range)(features))
        # There will be no division by zero here unless the WB range lower bound is 0
        # normalize by luminance
        color_scaling =color_scaling * (1.0 / (
            1e-5 + 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] +
            0.06 * color_scaling[:, 2])[:, None]) # (B,3)
        
        if torch.isnan(features).any():
            print("NaN occured in ImprovedWB ")
            return features
        if torch.isnan(color_scaling).any():
            print("NaN occured in ImprovedWB color ")
            return features
        # print(color_scaling.shape)
        return color_scaling

    def process(self, img, param, defog, IcA):
        
        return img * param[:, :, None, None]
        # return img 
        
class GammaFilter(Filter):  #gamma_param is in [-gamma_range, gamma_range]

    def __init__(self, cfg):
        super().__init__(cfg)
        self.short_name = 'G'
        self.begin_filter_parameter = cfg.gamma_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        log_gamma_range = torch.log(torch.tensor(self.cfg.gamma_range))
        return torch.exp(tanh_range(-log_gamma_range, log_gamma_range)(features))

    def process(self, img, param, defog_A, IcA):
        # print("para shape",param.shape)
        param_1 = param.repeat(1,3)
        if torch.isnan(param_1).any():
            print("NaN occured in Gamma ")
            return img
        return torch.pow(torch.maximum(img, torch.tensor(0.0001)), param_1[:, :, None, None])
        # return img
            
class ToneFilter(Filter):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.curve_steps = cfg.curve_steps
        self.short_name = 'T'
        self.begin_filter_parameter = cfg.tone_begin_param

        self.num_filter_parameters = cfg.curve_steps

    def filter_param_regressor(self, features):
        # features (B,cureve_step)
        tone_curve = features.view(-1, 1, 1, 1, self.cfg.curve_steps) # (B,1,1,1,8)
        tone_curve = tanh_range(*self.cfg.tone_curve_range)(tone_curve)
        return tone_curve

    def process(self, img, param, defog, IcA):

        tone_curve = param
        tone_curve_sum = tone_curve.sum(dim=4, keepdim=True).squeeze(4) + 1e-30
        if torch.isnan(tone_curve_sum).any():
            print("NaN occured in tone_curve_sum ")
            return img
        # print(tone_curve_sum.shape) # (B,1,1,1)
        total_image = img * 0
        for i in range(self.cfg.curve_steps):
            weight = param[..., i]  # (batch_size, 1, 1, 1)
            tone_mask = torch.clamp(img - (1.0 * i / self.cfg.curve_steps), 0, 1.0 / self.cfg.curve_steps)
            # print("mask shape",tone_mask.shape)
            total_image += tone_mask * weight  # 权重加权
        if torch.isnan(total_image).any():
            print("NaN occured in total_image ")
            return img
        # print(total_image.shape)
        total_image *= self.cfg.curve_steps / tone_curve_sum
        if torch.isnan(total_image).any():
            print("NaN occured in  2nd total_image ")
            return img
        img = total_image
        return img

class ContrastFilter(Filter):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.short_name = 'Ct'
        self.begin_filter_parameter = cfg.contrast_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
    # return tf.sigmoid(features)
        return torch.tanh(features)

    def process(self, img, param, defog, IcA):
        luminance = torch.minimum(torch.maximum(rgb2lum(img), torch.tensor(0.0)), torch.tensor(1.0)) #(B,1,H,W)
        #print(luminance.shape)
        contrast_lum = -torch.cos(torch.math.pi * luminance) * 0.5 + 0.5
        if torch.isnan(contrast_lum).any():
            print("NaN occured in contrast_lum ")
            return img
        contrast_image = img / (luminance + 1e-6) * contrast_lum
        if torch.isnan(contrast_image).any():
            print("NaN occured in contrast_image ")
            return img
        img_mo = lerp(img, contrast_image, param[:, :, None, None])
        if torch.isnan(img_mo).any():
            print("NaN occured in contrast ")
            return img
        return img_mo

class UsmFilter(Filter):#Usm_param is in [Defog_range]

    def __init__(self, cfg):
        super().__init__(cfg)
        self.short_name = 'UF'
        self.begin_filter_parameter = cfg.usm_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.usm_range)(features)

    def process(self, img, param, defog_A, IcA):
        B, C, H, W = img.shape
        def make_gaussian_2d_kernel(sigma, dtype=torch.float32):
            radius = 12
            x = torch.arange(-radius, radius + 1, dtype=dtype)
            k = torch.exp(-0.5 * (x / sigma)**2)
            k = k / k.sum()
            
            kernel_2d = torch.outer(k, k)
            kernel_2d = kernel_2d.view(1, 1, 25, 25)  # (1,1,H,W)
            
            
            return kernel_2d 
    

        kernel_i = make_gaussian_2d_kernel(5)
        kernel_i = kernel_i.repeat(B*C, 1, 1, 1).to(img.device)
        
        if torch.isnan(kernel_i).any():
                print("NaN occured in usm kernel_2d ")
                return img
        # print('kernel_i.shape', kernel_i.shape)
        

        pad_w = (25 - 1) // 2
        padded = F.pad(img, (pad_w, pad_w, pad_w, pad_w), mode='reflect') #(B,C,H',W')
        
        blurred = F.conv2d(
            input=padded.view(1, B*C, H + 2*pad_w, W + 2*pad_w),
            weight=kernel_i,
            bias=None,
            stride=1,
            padding=0,
            groups=B*C  
        )
        if torch.isnan(blurred).any():
            print("NaN occured in usm blurred ")
            return img
        # print(blurred.shape)
        blurred = blurred.view(B, C, H, W)
        # print(blurred.shape)
        if torch.isnan(param).any():
            print("NaN occured in usm param ")
            return img
        img_out = (img - blurred) * param[:, :, None, None] + img
        
        if torch.isnan(img_out).any():
            print("NaN occured in usm img_out ")
            return img
        # img_out = (img - output) * 2.5 + img
        return img_out
    
if __name__ == '__main__':
    #filter = DefogFilter(cfg)
    #filter = ImprovedWhiteBalanceFilter(cfg)
    #filter = GammaFilter(cfg)
    #filter = ContrastFilter(cfg)
    #filter = UsmFilter(cfg)
    # filter = ToneFilter(cfg)
    filtered_image_batch = torch.ones((10,3,16,16))
    filter_features = torch.ones((10,15))
    defog_A = torch.zeros((10, 3))
    IcA = torch.zeros((10,16, 16))
                    
    filter.apply(filtered_image_batch, filter_features, defog_A, IcA)
    