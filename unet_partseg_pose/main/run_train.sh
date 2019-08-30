#!/usr/bin/env/bash
CUDA_VISIBLE_DEVICES=$1 python train_model.py \
    --model_folder_name unet_adam1e-4_scratch_B1 \
    --TFR /media/data_cifs/lakshmi/MGH/mgh-pose/mgh_train_bootstrap_v1.tfrecords \
    --common_path /media/data_cifs/lakshmi/project-worms/mgh/ \
    --input_size 448 \
    --batch_size 32 \
    --num_classes 8
#    --single_channel
#--weight_decay
