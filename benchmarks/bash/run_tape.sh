#!/bin/bash

level="2a"
if [ "$level" = "1" ]; then
    pep_max_len=15
    tcr_max_len=19
elif [ "$level" = "2a" ] || [ "$level" = "2a_basic" ]; then
    pep_max_len=44
    tcr_max_len=19
elif [ "$level" = "2b" ] || [ "$level" = "2b_basic" ] || [ "$level" = "3" ] || [ "$level" = "3_basic" ]; then
    pep_max_len=10
    tcr_max_len=19
elif [ "$level" = "4" ] || [ "$level" = "4_basic" ]; then
    pep_max_len=10
    tcr_max_len=121
else
    echo "Invalid level value"
fi
echo "pep_max_len: $pep_max_len, tcr_max_len: $tcr_max_len"

model_type="tape"
lr=3e-6
log_path="/home/lujd/Hi-TpH/benchmarks/new_logs/${model_type}/"
mkdir -p "$log_path"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
        --nproc_per_node=4 \
        --master_port=25661 \
        /home/lujd/Hi-TpH/benchmarks/train_main.py \
        --epochs 100 \
        --rand_neg True \
        --lr $lr \
        --batch_size 32 \
        --plm tape \
        --finetune \
        --data_path "/data/lujd/TCRdata/benchmarks/level${level}/" \
        --model_path "/data/lujd/TCRmodel/benchmarks/level${level}/${model_type}/" \
        --pep_max_len $pep_max_len \
        --tcr_max_len $tcr_max_len \
        --level "$level" \
        --early_stop 5 \
        > "${log_path}/level${level}_${lr}.txt"
