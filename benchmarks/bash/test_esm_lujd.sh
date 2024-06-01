#!/bin/bash

level="1"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
        --nproc_per_node=2 \
        --master_port=25646 \
        /home/lujd/Hi-TpH/benchmarks/test_main.py \
        --rand_neg True \
        --batch_size 128 \
        --data_path "/data/lujd/TCRdata/benchmarks/level${level}/" \
        --model_path "/data/lujd/TCRmodel/benchmarks/level${level}/esm/" \
        --checkpoint_file "plm_esm_Finetune_B128_LR8e-05_seq_cat_fold0_ep24_240601.pkl" \
        --pep_max_len 15 \
        --tcr_max_len 19 \
        --level "$level" \
        >> "/home/lujd/Hi-TpH/benchmarks/new_logs/esm/test_level${level}.txt"


# 1:  15 19
# 2a: 44 19     # 2a_basic
# 2b: 10 19     # 2b_basic
# 3:  10 19     # 3_basic
# 4:  10 121    # 4_basic

# 3 8e-5 5e-5