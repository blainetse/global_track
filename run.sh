python train_qg_rcnn.py --config configs/qg_rcnn_r50_fpn.py --load_from checkpoints/qg_rcnn_r50_fpn_2x_20181010-443129e1.pth --gpus 1


# python -m torch.distributed.launch --nproc_per_node=4 train_qg_rcnn.py \
#     --launcher pytorch --load_from checkpoints/qg_rcnn_r50_fpn_2x_20181010-443129e1.pth \
#     --base_dataset "coco_train,got10k_train,lasot_train" --sampling_prob "0.4,0.4,0.2" \
#     --gpus 2 --work_dir checkpoints/qg_rcnn_r50_fpn_coco_got10k_lasot
