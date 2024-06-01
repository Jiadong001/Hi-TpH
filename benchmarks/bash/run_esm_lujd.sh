#!/bin/bash

level="4"
CUDA_VISIBLE_DEVICES=0,1,3,4 python -m torch.distributed.run \
        --nproc_per_node=4 \
        --master_port=25646 \
        /home/lujd/Hi-TpH/benchmarks/train_main.py \
        --epochs 60 \
        --rand_neg True \
        --lr 8e-5 \
        --batch_size 16 \
        --plm esm \
        --finetune \
        --model_path "/data/lujd/TCRmodel/level${level}/esm/" \
        --data_path "/data/lujd/TCRdata/level${level}/" \
        --pep_max_len 9 \
        --tcr_max_len 121 \
        --level "$level" \
        --early_stop 5 \
        > "/home/lujd/Hi-TpH/benchmarks/logs/esm/level${level}.txt"


# 1:  15 19
# 2a: 43 19     # 2a_basic

# 2b(old version): 9 34
# 2b: 9 19      # 2b_basic

# 3:  9 19      # 3_basic
# 4:  9 121     # 4_basic