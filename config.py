#! /usr/bin/env python
# coding=utf-8


from easydict import EasyDict as edict
from filters import *




__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

###########################################################################
# Filter Parameters
###########################################################################


cfg.filters = [
    DefogFilter, ImprovedWhiteBalanceFilter,  GammaFilter,
    ToneFilter, ContrastFilter, UsmFilter
]
cfg.num_filter_parameters = 15

cfg.defog_begin_param = 0

cfg.wb_begin_param = 1
cfg.gamma_begin_param = 4
cfg.tone_begin_param = 5
cfg.contrast_begin_param = 13
cfg.usm_begin_param = 14


cfg.curve_steps = 8
cfg.gamma_range = 3
cfg.exposure_range = 3.5
cfg.wb_range = 1.1
cfg.color_curve_range = (0.90, 1.10)
cfg.lab_curve_range = (0.90, 1.10)
cfg.tone_curve_range = (0.5, 2)
cfg.defog_range = (0.1, 1.0)
cfg.usm_range = (0.0, 5)

cfg.archi_path1 = 'model_cfg/training/yolov7.yaml'
cfg.archi_path2 = 'model_cfg/training/yolov7_Simam.yaml'
cfg.add_simam_attention = False
cfg.e_lambda = 1e-4   # regularization
cfg.dropout_rate = 0.2
# Masking is DISABLED
cfg.masking = False
cfg.minimum_strength = 0.3
cfg.maximum_sharpness = 1
cfg.clamp = False

###########################################################################
# CNN Parameters
###########################################################################
cfg.source_img_size = 64
cfg.base_channels = 32
cfg.dropout_keep_prob = 0.5
# G and C use the same feed dict?
cfg.share_feed_dict = True
cfg.shared_feature_extractor = True
cfg.fc1_size = 128
cfg.bnw = False
# number of filters for the first convolutional layers for all networks
#                      (stochastic/deterministic policy, critic, value)
cfg.feature_extractor_dims = 4096
