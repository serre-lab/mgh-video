#!/usr/bin/env/bash
python train_model.py \
    --use_tpu True \
    --tpu l-v3-8-1 \
    --tpu_zone us-central1-a \
    --train_num_cores 8 \
    --eval_tpu r-12 \
    --eval_tpu_zone us-central1-f \
    --eval_num_cores 8 \
    --gcp_project beyond-dl-1503610372419 \
    --data_dir gs://serrelab/MGH/tfrecords/v1_selected_pretrainedi3d_uniformsample/mgh_train_directory \
    --model_dir gs://serrelab/MGH/model_runs/v1_uniformsample_v3-8_b256_15classes_adamlre-3_i3d_weightedloss_logits \
    --profile_every_n_steps 0 \
    --mode train \
    --train_steps 6000 \
    --train_batch_size 256 \
    --eval_batch_size 192 \
    --num_train_videos 4500 \
    --num_eval_videos 18500 \
    --num_classes 15 \
    --steps_per_eval 1251 \
    --iterations_per_loop 300 \
    --num_parallel_calls 8 \
    --precision float32 \
    --optimizer adam \
    --base_learning_rate 1e-3 \
    --momentum 0.9 \
    --weight_decay 1e-5 \
    --label_smoothing 0.0 \
    --log_step_count_steps 64 \
    --use_batch_norm \
    --use_cross_replica_batch_norm \
    #--init_checkpoint None
    #--skip_host_call \
    #--use_cache \
    #--export_to_tpu \
    #--enable_lars \
    #--use_async_checkpointing \
    "$@"
