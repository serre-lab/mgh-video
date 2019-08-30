#!/usr/bin/env/bash
python shard_creation.py \
    --frames_per_clip 32 \
    --phase mgh_train \
    --num_shards 256 \
    --frame_height 256 \
    --frame_width 256 \
    --channels 3 \
    --multithread 1 \
    --transition 0 \
    --clips_per_behavior 70000 \
    "$@"
