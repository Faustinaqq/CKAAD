#!/bin/bash
labeled_anomaly_ratio=0.05
epochs=10
num_classes=4
labeled_anomaly_class_num=1
loops=$((num_classes-1))
for j in $(seq 1 $loops)
do
        CUDA_VISIBLE_DEVICES=0 python main.py --dataset oct --batch_size 32 --lr 1e-03 --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
        --normal 0 --seed 111 --labeled_anomaly_class_num ${labeled_anomaly_class_num} --labeled_anomaly_class $j --img_size 256 \
        --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
        --log_dir ./log --model resnet18 --layer 2 3
done