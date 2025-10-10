#!/bin/bash

# level="1"
# components="pep beta"
# pep_max_len=15
# tcr_max_len=19

# level="2"
# components="pep hla beta"
# pep_max_len=44
# tcr_max_len=19

level="3"
components="pep hla ab"
pep_max_len=10
tcr_max_len=19

# level="4"
# components="pep hla ab"
# pep_max_len=10
# tcr_max_len=121

echo "pep_max_len: $pep_max_len, tcr_max_len: $tcr_max_len"
esm_type="esm2-150M"
lr=8e-5
batch_size=16
model_path="/data/lujd/TCRmodel/immrep/level${level}/${esm_type}/" 
log_path="../logs/train_immrep/${esm_type}/"
mkdir -p "$log_path"

CUDA_VISIBLE_DEVICES=0,3,5,6 python -m torch.distributed.run \
        --nproc_per_node=4 \
        --master_port=7776 \
        ../train_main.py \
        --epochs 100 \
        --rand_neg \
        --lr $lr \
        --batch_size $batch_size \
        --plm "$esm_type" \
        --finetune \
        --data_path "../../benchmarks_dataset/level${level}/" \
        --model_path "$model_path" \
        --test_data immrep2023 \
        --pep_max_len $pep_max_len \
        --tcr_max_len $tcr_max_len \
        --level "$level" \
        --components "$components" \
        --early_stop 5 \
        > "${log_path}/level${level}_${lr}_${batch_size}.txt"
