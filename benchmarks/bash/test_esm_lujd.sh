#!/bin/bash

level="4"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
        --nproc_per_node=2 \
        --master_port=25646 \
        /home/lujd/Hi-TpH/benchmarks/test_main.py \
        --rand_neg True \
        --batch_size 128 \
        --data_path "/data/lujd/TCRdata/level${level}/" \
        --model_path "/data/lujd/TCRmodel/level${level}/esm/" \
        --checkpoint_file "plm_esm_Finetune_B16_LR0.0001_seq_cat_fold0_ep12_240601.pkl" \
        --pep_max_len 9 \
        --tcr_max_len 121 \
        --level "$level" \
        >> "/home/lujd/Hi-TpH/benchmarks/logs/esm/test_level${level}.txt"


# 1:  15 19
# 2a: 43 19     # 2a_basic

# 2b(old version): 9 34
# 2b: 9 19      # 2b_basic

# 3:  9 19      # 3_basic
# 4:  9 121     # 4_basic

# 3 8e-5 5e-5