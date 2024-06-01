#!/bin/bash
CUDA_VISIBLE_DEVICES=3,5 python -m torch.distributed.run \
        --nproc_per_node=2 \
        --master_port=25641 \
        /home/luyq/code/ProteinLanguageModels/src/tcr_pmhc/train_main.py \
        --epochs 30 \
        --rand_neg True \
        --lr 1e-5 \
        --batch_size 14 \
        --plm tape \
        --pep_max_len 43 \
        --tcr_max_len 19 \
        --model_path /data/luyq/level2b/ \
        --data_path /data/luyq/level2b/ \
        2>&1 | tee /home/luyq/logs/train-test.txt
