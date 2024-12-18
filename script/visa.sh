#!/bin/bash
labeled_anomaly_ratio=0.05
labeled_anomaly_class_num=1
epochs=400
eval_epoch=1
for normal in 'capsules'
do
    echo $normal
    CUDA_VISIBLE_DEVICES=1 python main.py --dataset visa --batch_size 16 \
     --lr 5e-03 --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
    --normal $normal --seed 111 --img_size 256 \
    --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    --labeled_anomaly_class 0 \
    --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    --log_dir ./log --model wide_resnet50_2 --eval_epoch ${eval_epoch} --layer 1 2 3
done