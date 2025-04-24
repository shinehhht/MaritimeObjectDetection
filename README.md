# 满足三级海况、水平能见度不大于500米的雾天和夜间使用要求 

## Dataset
[Seadronessee](https://seadronessee.cs.uni-tuebingen.de/)
- 原始 dataset: <code>data/train_formal</code>, <code>data/val_formal</code>
- 增雾 dataset: <code>data/train_with_fog</code>, <code>data/val_with_fog</code>
- 混合 dataset(以1:4比例组合训练集未增雾和增雾): <code>data/train_with_mixed</code>
- 增雾脚本: <code>/data/foggy_data_make.py</code>

[Visible-Thermal](https://arxiv.org/abs/2406.14482)
- data/VTT

## Training
To train Image-Adaptive-YOLO
```
python train.py --workers 8 --device 1 --batch-size 8 --data data/foggy.yaml --img 640 640 --cfg model_cfg/training/yolov7.yaml --weights 'yolov7_training.pt' --name image-adaptive-yolo --hyp data/hyp.scratch.p5.yaml --fog_FLAG --ISP_FLAG
```

To train Image-Adaptive-Fuse-YOLO 
```
python train_fuse.py --workers 8 --device 1 --batch-size 8 --data data/foggy_fuse.yaml --img 640 640 --cfg model_cfg/training/yolov7.yaml --weights 'yolov7_training.pt' --name image-adaptive-fuse-yolo --hyp data/hyp.scratch.p5.yaml --fog_FLAG --ISP_FLAG
```

To train Feature-Adaption-YOLO
```
python train_adaption.py --workers 8 --device 1 --batch-size 8 --data data/foggy_fuse.yaml --img 640 640 --cfg model_cfg/training/yolov7.yaml --weights 'yolov7_training.pt' --name feature-adaption-yolo --hyp data/hyp.scratch.p5.yaml
```

To train Cross-Modality-Fusion-YOLO
```
python train_multimodality.py --workers 8 --device 1 --batch-size 8 --data data/multimodal.yaml --img 640 640  --cfg model_cfg/training/yolov7-transformerx3.yaml --weights 'yolov7_training.pt' --name multimodality-yolo --hyp data/hyp.scratch.p5.yaml
```



