#!/bin/bash

level="1"
CUDA_VISIBLE_DEVICES=0,1,6 python -m torch.distributed.run \
        --nproc_per_node=3 \
        --master_port=25646 \
        /home/lujd/Hi-TpH/benchmarks/train_main.py \
        --epochs 100 \
        --rand_neg True \
        --lr 8e-5 \
        --batch_size 128 \
        --plm esm \
        --model_path "/data/lujd/TCRmodel/benchmarks/level${level}/esm_frozen/" \
        --data_path "/data/lujd/TCRdata/benchmarks/level${level}/" \
        --pep_max_len 15 \
        --tcr_max_len 19 \
        --level "$level" \
        --early_stop 5 \
        > "/home/lujd/Hi-TpH/benchmarks/new_logs/esm_frozen/level${level}.txt"


# 1:  15 19
# 2a: 44 19     # 2a_basic
# 2b: 10 19     # 2b_basic
# 3:  10 19     # 3_basic
# 4:  10 121    # 4_basic
        # --finetune \
