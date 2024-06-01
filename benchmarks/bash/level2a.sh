#!/bin/bash
CUDA_VISIBLE_DEVICES=2,5,7 python -m torch.distributed.run \
        --nproc_per_node=3 \
        --master_port=25647 \
        /home/luyq/code/ProteinLanguageModels/src/tcr_pmhc/train_main.py \
        --epochs 30 \
        --rand_neg True \
        --lr 1e-5 \
        --batch_size 14 \
        --plm esm \
        --model_path /data/luyq/level2a/ \
        --data_path /data/luyq/level2a/ \
        --pep_max_len 43 \
        --tcr_max_len 19 \
        --level 2a \
        --checkpoint_file /data/luyq/level2a/plm_tape_Finetune_B14_LR1e-05_seq_cat_fold0_ep14_240529.pkl \
        --early_stop 10 \
        2>&1 | tee /home/luyq/logs/train-test.txt
