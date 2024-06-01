#!/bin/bash

level="2a_basic"
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.run \
        --nproc_per_node=1 \
        --master_port=25642 \
        /home/lujd/Hi-TpH/benchmarks/train_main.py \
        --epochs 100 \
        --rand_neg True \
        --lr 1e-3 \
        --batch_size 1024 \
        --plm baseline_mlp \
        --model_path "/data/lujd/TCRmodel/benchmarks/level${level}/mlp" \
        --data_path "/data/lujd/TCRdata/benchmarks/level${level}/" \
        --pep_max_len 44 \
        --tcr_max_len 19 \
        --level "$level" \
        --early_stop 5 \
        > "/home/lujd/Hi-TpH/benchmarks/new_logs/mlp/level${level}.txt"


# 1:  15 19
# 2a: 44 19     # 2a_basic
# 2b: 10 19     # 2b_basic
# 3:  10 19     # 3_basic
# 4:  10 121    # 4_basic