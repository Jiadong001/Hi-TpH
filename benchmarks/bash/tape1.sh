#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,3,4 python -m torch.distributed.run \
        --nproc_per_node=4 \
        --master_port=25642 \
        /home/luyq/code/ProteinLanguageModels/src/tcr_pmhc/train_main.py \
        --epochs 50 \
        --rand_neg True \
        --lr 1e-5 \
        --batch_size 14 \
        --data_path /data/luyq \
        --model_path /data/luyq \
        --pep_max_len 15 \
        --tcr_max_len 20 \
        --plm tape \
        --level 1 \
        --checkpoint_file /data/luyq/plm_tape_Finetune_B14_LR1e-05_seq_cat_fold0_ep30_240529.pkl \
        2>&1 | tee /home/luyq/logs/train-test.txt
