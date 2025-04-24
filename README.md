# 满足三级海况、水平能见度不大于500米的雾天和夜间使用要求 

## Dataset
[Seadronessee](https://seadronessee.cs.uni-tuebingen.de/)
- 原始 dataset: <code>data/train_formal</code>, <code>data/val_formal</code>
- 增雾 dataset: <code>data/train_with_fog</code>, <code>data/val_with_fog</code>
- 混合 dataset(以1:4比例组合训练集未增雾和增雾): <code>data/train_with_mixed</code>
- 增雾脚本: <code>/data/foggy_data_make.py</code>

[Visible-Thermal](https://arxiv.org/abs/2406.14482)
- data/VTT

## 整体模型框架
### [Image-Adaptive-YOLO](https://arxiv.org/abs/2112.08088)
- train: <code>train.py</code>
- test: <code>test.py</code>
- 模型模块: 
    - 整体模型 <code>models/combination_model.py</code>
    - 小型CNN <code>models/CNNPP.py</code>
    - YOLO <code>models/yolo.py</code>
    - Filters <code>filters.py</code>
- [SimAm](https://proceedings.mlr.press/v139/yang21o) 注意力模块: 
    - <code>--Use_Simam_YOLO</CODE>在YOLO结构上插入一些注意力模块 
    - 更改后yolo结构: <code>model_cfg/training/yolov7_Simam.yaml</code>
    - SimAm模块: <code>models/simam_module.py</code>

### Image-Adaptive-Fuse-YOLO 
借鉴Feature-Adaption-YOLO，设计一个特征融合模块，把经过filter处理的image feature和原未增过雾的image feature
- train: <code>train_fuse.py</code>
- test: <code>test_fuse.py</code>
- 模型模块:
    - 整体模型 <code>models/combination_model_fuse.py</code>
    - 提取原始图像feature <code>models/Darknet53.py</code>
    - 特征融合模块 <code>models/feature_fusion.py</code>
    - YOLO <code>models/yolo_fuse.py</code>

### [Feature-Adaption-YOLO](https://arxiv.org/pdf/2403.09233)
- train: <code>train_adaption.py</code>
- test: <code>test_adaption.py</code>
- 模型模块:
    - 整体模型 <code>models/combination_model_adaption.py</code>
    - 提取原始图像feature <code>models/Darknet53.py</code>
    - feature adaption module <code>models/feature_adaption.py</code>
    - YOLO <code>models/yolo_adaption.py</code>

### [Cross-Modality Fusion YOLO](https://arxiv.org/abs/2111.00273)
- train: <code>train_multimodality.py</code>
- test: <code>test_multimodality.py</code>
- 模型模块:
    - 整体模型 <code>models/yolo_multimodality.py</code>
    - 模型结构 <code>model_cfg/training/yolov7-transformerx3.yaml</code>

### Cross-Modality SwinFusion YOLO
借鉴[SwinFusion:Cross-domain Long-range Learning for General Image Fusion via Swin Transformer](https://ieeexplore.ieee.org/document/9812535) 在模态融合时利用swin fusion
- train: <code>train_multimodality.py</code>
- test: <code>test_multimodality.py</code>
- 模型模块:
    - 整体模型 <code>models/yolo_multimodality.py</code>
    - 模型结构 <code>model_cfg/training/yolov7-transformerx3_swin.yaml</code>
    - Swin Fusion模块 <code>models/swinfusion.py</code>

