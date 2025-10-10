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
checkpoint="plm_esm2-150M_Finetune_B16_LR8e-05_seq_cat_fold0_ep6_251010.pkl"
model_path="/data/lujd/TCRmodel/immrep/level${level}/${esm_type}/"
log_path="../logs/eval_immrep/${esm_type}/"
mkdir -p "$log_path"

CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.run \
        --nproc_per_node=1 \
        --master_port=6666 \
        ../test_main.py \
        --batch_size 128 \
        --data_path "../../benchmarks_dataset/level${level}/" \
        --model_path "$model_path" \
        --test_data immrep2023 \
        --checkpoint_file "$checkpoint" \
        --pep_max_len $pep_max_len \
        --tcr_max_len $tcr_max_len \
        --level "$level" \
        --components "$components" \
        >> "${log_path}/test_level${level}.txt"
