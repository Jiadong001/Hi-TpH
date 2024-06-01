#!/bin/bash

level="2b_basic"
CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.run \
        --nproc_per_node=1 \
        --master_port=25646 \
        /home/lujd/Hi-TpH/benchmarks/train_main.py \
        --epochs 100 \
        --rand_neg True \
        --lr 1e-3 \
        --batch_size 1024 \
        --plm baseline_mlp \
        --model_path "/data/lujd/TCRmodel/level${level}/" \
        --data_path "/data/lujd/TCRdata/level${level}/" \
        --pep_max_len 9 \
        --tcr_max_len 19 \
        --level "$level" \
        --early_stop 5 \
        > "/home/lujd/Hi-TpH/benchmarks/logs/mlp/mlp_level${level}.txt"


# 1:  15 19
# 2a: 43 19     # 2a_basic

# 2b(old version): 9 34
# 2b: 9 19      # 2b_basic

# 3:  9 19      # 3_basic
# 4:  9 121     # 4_basic