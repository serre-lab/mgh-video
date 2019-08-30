#!/usr/bin/env/bash
python train_model.py \
    --use_tpu True \
    --tpu l-v3-8-3 \
    --tpu_zone us-central1-a \
    --train_num_cores 8 \
    --eval_tpu r-12 \
    --eval_tpu_zone us-central1-f \
    --eval_num_cores 8 \
    --gcp_project beyond-dl-1503610372419 \
    --data_dir gs://serrelab/MGH/tfrecords/v2_selected_pretrainedi3d_chunks_32seq_combined/mgh_train_directory \
    --model_dir gs://serrelab/MGH/model_runs/v2_chunks_32seq_combined_v3-8_b256_adamlre-3_i3d_weightedloss_earlyendpoint_block+logits \
    --profile_every_n_steps 0 \
    --mode train \
    --train_steps 14000 \
    --train_batch_size 256 \
    --eval_batch_size 192 \
    --num_train_videos 4500 \
    --num_eval_videos 18500 \
    --num_classes 11 \
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
    --final_endpoint Mixed_5b \
    --warm_start_vars Cond3d_*w,Conv3d_*beta,Mixed_3*w,Mixed_3*beta,Mixed_4*w,Mixed_4*beta \
    --optimize_var_scopes Mixed_5b, Logits \
    --time_divisor 2 \
    --hw_divisor 7 \
    #--init_checkpoint None
    #--skip_host_call \
    #--use_cache \
    #--export_to_tpu \
    #--enable_lars \
    #--use_async_checkpointing \
    "$@"
