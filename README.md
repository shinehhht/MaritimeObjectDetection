# ELEG 5491 Project Foggy maritime target detection 

## Dataset
[Seadronessee](https://seadronessee.cs.uni-tuebingen.de/)
- origin dataset: <code>data/train_formal</code>, <code>data/val_formal</code>
- foggy dataset: <code>data/train_with_fog</code>, <code>data/val_with_fog</code>
- mixed dataset(Combining the training set un-fogged and fogged in a 1:4 ratio): <code>data/train_with_mixed</code>
- Image fogging script: <code>/data/foggy_data_make.py</code>

[Visible-Thermal](https://arxiv.org/abs/2406.14482)
- data/VTT

The processed dataset should conform to the following structure
```
/data
    |── train_formal/
    |           |—— images
    |           |—— labels
    |── train_with_fog/
    |           |—— images
    |           |—— labels
    |── val_formal/
    |           |—— images
    |           |—— labels
    |── val_with_fog/
    |           |—— images
    |           |—— labels
    |── train_with_mixed/
    |           |—— images
    |           |—— labels
    |── VTT/
    |     |—— train_00/
    |     |        |—— images
    |     |        |—— labels
    |     |—— train_01/
    |     |        |—— images
    |     |        |—— labels
    |     |—— test_00/
    |     |        |—— images
    |     |        |—— labels
    |     |—— test_01/
    |     |        |—— images
    |     |        |—— labels
    |—— foggy_data_make.py
    |—— foggy.yaml
    |—— foggy_fuse.yaml
    |—— multimodal.yaml
    

```
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



