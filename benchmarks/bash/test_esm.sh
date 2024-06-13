#!/bin/bash

level="4_four"
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
elif [ "$level" = "4_one" ] || [ "$level" = "4_twoa" ] || [ "$level" = "4_twob" ] || [ "$level" = "4_three" ] || [ "$level" = "4_four" ]; then
    pep_max_len=10
    tcr_max_len=121
else
    echo "Invalid level value"
fi
echo "pep_max_len: $pep_max_len, tcr_max_len: $tcr_max_len"

esm_type="esm35_supp"
checkpoint="plm_esm-35M_Finetune_B32_LR6e-05_seq_cat_fold0_ep7_240610.pkl"
log_path="/home/lujd/Hi-TpH/benchmarks/new_logs/${esm_type}/"
mkdir -p "$log_path"

CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.run \
        --nproc_per_node=1 \
        --master_port=25641 \
        /home/lujd/Hi-TpH/benchmarks/test_main.py \
        --rand_neg True \
        --batch_size 128 \
        --data_path "/data/lujd/TCRdata/benchmarks/level${level}/" \
        --model_path "/data/lujd/TCRmodel/benchmarks/level${level}/${esm_type}/" \
        --checkpoint_file "$checkpoint" \
        --pep_max_len $pep_max_len \
        --tcr_max_len $tcr_max_len \
        --level "$level" \
        >> "${log_path}/test_level${level}.txt"
