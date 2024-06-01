#!/bin/bash
CUDA_VISIBLE_DEVICES=1,3,7 python -m torch.distributed.run \
        --nproc_per_node=3 \
        --master_port=25645 \
        /home/luyq/code/ProteinLanguageModels/src/tcr_pmhc/train_main.py \
        --epochs 60 \
        --rand_neg True \
        --lr 1e-4 \
        --batch_size 14 \
        --data_path /data/luyq \
        --model_path /data/luyq/esm_pkl \
        --plm esm \
        --level 1 \
        --tcr_max_len 20 \
        --pep_max_len 15 \
        --early_stop 10 \
        2>&1 | tee /home/luyq/logs/train-test.txt
