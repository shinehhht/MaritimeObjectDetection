# echo "train yolo_with_filter"
# python train.py --workers 8 --device 1 --batch-size 8 --data data/foggy.yaml --img 640 640 --cfg model_cfg/training/yolov7.yaml --weights 'yolov7_training.pt' --name yolov7_hybrid_origin --hyp data/hyp.scratch.p5.yaml

# python train_yolov7.py --workers 8 --device 0 --batch-size 8 --data foggy.yaml --img 640 640 --cfg model_cfg/training/yolov7.yaml --weights 'yolov7_training.pt' --name yolov7 --hyp data/hyp.scratch.p5.yaml

python train_multimodality.py --workers 8 --device 1 --batch-size 1 --data foggy.yaml --img 640 640  --cfg model_cfg/training/yolov7-transformerx3_swin.yaml --weights 'yolov7_training.pt' --name yolov7_filter_fuse_foggy_pretrain_ --hyp data/hyp.scratch.p5.yaml #--fog_FLAG --ISP_FLAG # --Use_Simam_YOLO
# python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py --workers 4 --device 0,1 --sync-bn --batch-size 8 --data foggy.yaml --img 640 640 --cfg model_cfg/training/yolov7.yaml --weights 'yolov7_training.pt' --name yolov7_filter_foggy_pretrain --hyp data/hyp.scratch.p5.yaml
