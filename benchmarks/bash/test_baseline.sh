#!/bin/bash

level="1"
CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.run \
        --nproc_per_node=1 \
        --master_port=25646 \
        /home/lujd/Hi-TpH/benchmarks/test_main.py \
        --rand_neg True \
        --batch_size 1024 \
        --data_path "/data/lujd/TCRdata/level${level}/" \
        --model_path "/data/lujd/TCRmodel/level${level}/" \
        --checkpoint_file "plm_baseline_mlp_WithoutFinetune_B1024_LR0.001_seq_cat_fold0_ep95_240601.pkl" \
        --pep_max_len 15 \
        --tcr_max_len 19 \
        --level "$level" \
        >> "/home/lujd/Hi-TpH/benchmarks/logs/mlp/test_mlp_level${level}.txt"


# 1:  15 19
# 2a: 43 19     # 2a_basic

# 2b(old version): 9 34
# 2b: 9 19      # 2b_basic

# 3:  9 19      # 3_basic
# 4:  9 121     # 4_basic