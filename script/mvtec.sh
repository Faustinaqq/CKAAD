#!/bin/bash
labeled_anomaly_ratio=0.05
labeled_anomaly_class_num=1
for normal in 'bottle'
do
    echo $normal

    case $normal in 'carpet')
            epochs=10
            eval_epoch=1
            ;;
        *)  
            epochs=200
            eval_epoch=1
            ;;
    esac

    case $normal in 'transistor')
            lr=1e-03
            ;;
        *)  
            lr=5e-03
            ;;
    esac

    CUDA_VISIBLE_DEVICES=5 python main.py --dataset mvtec --batch_size 16 \
     --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
    --normal $normal --seed 111 --img_size 256 \
    --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    --labeled_anomaly_class 0 \
    --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    --log_dir ./log --model wide_resnet50_2 --eval_epoch ${eval_epoch} --layer 1 2 3
done