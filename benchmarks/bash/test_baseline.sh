#!/bin/bash

level="2a"
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
        --nproc_per_node=1 \
        --master_port=25643 \
        /home/lujd/Hi-TpH/benchmarks/test_main.py \
        --rand_neg True \
        --batch_size 1024 \
        --data_path "/data/lujd/TCRdata/benchmarks/level${level}/" \
        --model_path "/data/lujd/TCRmodel/benchmarks/level${level}/mlp" \
        --checkpoint_file "plm_baseline_mlp_WithoutFinetune_B1024_LR0.001_seq_cat_fold0_ep95_240601.pkl" \
        --pep_max_len 44 \
        --tcr_max_len 19 \
        --level "$level" \
        >> "/home/lujd/Hi-TpH/benchmarks/logs/mlp/test_mlp_level${level}.txt"


# 1:  15 19
# 2a: 44 19     # 2a_basic
# 2b: 10 19     # 2b_basic
# 3:  10 19     # 3_basic
# 4:  10 121    # 4_basic