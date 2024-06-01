import datetime
import os
import random
import time
from performances import performances

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from data_loader import tcr_pmhc_Dataset, tcr_pmhc_Dataset_RN, seq2token, data_loader
from torch.utils.data.distributed import DistributedSampler


# Set Seed
seed = 42
# Python & Numpy seed
random.seed(seed)
np.random.seed(seed)
# PyTorch seed
torch.manual_seed(seed)  # default generator
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# CUDNN seed
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def transfer(y_prob, threshold=0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])

def model_train_step(args, train_sampler, train_loader, model, epoch, device, local_rank):
    fold, plm_type, plm_input_type, epochs, l_r, threshold, rand_neg = (
        args.fold, args.plm, args.plm_input, args.epochs, args.lr,
        args.threshold, args.rand_neg)
    
    # set protbert tokenizer for baseline model
    plm_type = "protbert" if plm_type == "baseline_mlp" else plm_type
    
    train_sampler.set_epoch(seed + epoch - 1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=l_r)
    # optimizer = optim.AdamW(model.parameters(), lr=l_r)
    # print(optimizer)

    time_train_ep = 0
    model.train()

    y_true_train_list, y_prob_train_list, loss_train_list = [], [], []

    pbar = tqdm(train_loader)
    pbar.set_description(f"GPU{local_rank} Train epoch-{epoch}")

    for batch in pbar:
        if rand_neg:
            pos_tcr_list, neg_tcr_list, pep_seq_list = batch
            batch_num = len(pos_tcr_list)
            tcr_pmhc_tokens = seq2token(pos_tcr_list + neg_tcr_list,
                                    pep_seq_list + pep_seq_list,
                                    plm_type, plm_input_type, device)
            train_labels = [1] * batch_num + [0] * batch_num
            # if epoch==1:
            #     print(tcr_pmhc_tokens.shape)
        else:
            tcr_list, pep_seq_list, train_labels = batch
            batch_num = len(tcr_list)
            tcr_pmhc_tokens = seq2token(tcr_list, pep_seq_list, plm_type,
                                    plm_input_type, device)

        t1 = time.time()
        train_outputs = model(tcr_pmhc_tokens)

        y_true_train = torch.LongTensor(train_labels).to(device) if not rand_neg else torch.LongTensor([1] * batch_num +[0] *batch_num).to(device)
        train_loss = criterion(train_outputs, y_true_train)
        time_train_ep += time.time() - t1

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        y_prob_train = nn.Softmax(dim=1)(train_outputs)[:, 1].cpu().detach().numpy()
        y_true_train_list.extend(y_true_train.cpu().numpy())
        y_prob_train_list.extend(y_prob_train)
        loss_train_list.append(train_loss.item())

    y_pred_train_list = transfer(y_prob_train_list, threshold)
    ys_train = (y_true_train_list, y_pred_train_list, y_prob_train_list)

    ave_loss_train = sum(loss_train_list)/len(loss_train_list)
    print(
        "GPU{}-Fold-{}-RN-{}****Train (Ep avg): Epoch-{}/{} | Loss = {:.4f} | Time = {:.4f} sec"
        .format(local_rank, fold, rand_neg, epoch, epochs, ave_loss_train,
                time_train_ep))
    metrics_train = performances(y_true_train_list,y_pred_train_list,y_prob_train_list,print_=True)

    return ys_train, loss_train_list, metrics_train, time_train_ep

def make_validation(args, model, loader, device, local_rank):
    rand_neg, fold, plm_type, plm_input_type, threshold = args.rand_neg, args.fold, args.plm, args.plm_input, args.threshold

    # set protbert tokenizer for baseline model
    plm_type = "protbert" if plm_type == "baseline_mlp" else plm_type

    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_val_list, y_true_val_list, y_prob_val_list = [], [], []

    with torch.no_grad():
        pbar_desc = f"GPU{local_rank} VALIDATION {'without' if not rand_neg else 'with'} random negative samples"
        pbar = tqdm(loader, desc=pbar_desc)

        for batch in pbar:
            if not rand_neg:
                tcr_seq_list, pep_seq_list, val_labels = batch
                tcr_pmhc_tokens = seq2token(tcr_seq_list, pep_seq_list, plm_type,
                                        plm_input_type, device)
            else:
                pos_tcr_list, neg_tcr_list, pep_seq_list = batch
                batch_num = len(pos_tcr_list)
                tcr_pmhc_tokens = seq2token(
                    pos_tcr_list + neg_tcr_list,
                    pep_seq_list + pep_seq_list,
                    plm_type,
                    plm_input_type,
                    device,
                )
                val_labels = [1] * batch_num + [0] * batch_num  # pos + neg
                # print(tcr_pmhc_tokens.shape)

            val_outputs = model(tcr_pmhc_tokens)
            y_true_val = torch.LongTensor(val_labels).to(device)
            val_loss = criterion(val_outputs, y_true_val)

            y_prob_val = nn.Softmax(
                dim=1)(val_outputs)[:, 1].cpu().detach().numpy()
            y_true_val = np.array(val_labels)

            y_true_val_list.extend(y_true_val)
            y_prob_val_list.extend(y_prob_val)
            loss_val_list.append(val_loss.item())

        y_pred_val_list = transfer(y_prob_val_list, threshold)
        ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)

        ave_loss_val = sum(loss_val_list)/len(loss_val_list)
        print(
            f"GPU{local_rank}-Fold-{fold} ******{'VALIDATION'}****** : Loss = {ave_loss_val:.6f}"
        )
        metrics_val = performances(y_true_val_list,
                                   y_pred_val_list,
                                   y_prob_val_list,
                                   print_=True)

    return ys_val, ave_loss_val, metrics_val

def model_train(args,
                train_sampler,
                train_loader,
                val_loader,
                model,
                device,
                local_rank,
                world_size,
                start_epoch=0,
                validation_times=5):

    if not args.rand_neg:
        validation_times = 1

    valid_best, ep_best = -1, -1
    time_train = 0
    metrics_name = [
        "roc_auc", "accuracy", "mcc", "f1", "sensitivity", "specificity",
        "precision", "recall", "aupr"
    ]
    performance_val_df = pd.DataFrame()

    epoch, end_epoch = start_epoch, args.epochs

    train_sampler.set_epoch(seed + epoch - 1)
    while epoch < end_epoch:
        epoch += 1

        if local_rank == 0:
            print(
                f"Epoch:{epoch}***************************************************************************\n"
            )
        # 1. Train on GPUs
        ys_train, loss_train, metrics_train, time_train_ep = model_train_step(
            args, train_sampler, train_loader, model, epoch, device,
            local_rank)
        loss_train_tensor = torch.tensor(loss_train, device=device).sum()

        pass_tensor = loss_train_tensor.clone().detach()
        
        # 2. Train loss synchronization
        dist.barrier()
        dist.all_reduce(pass_tensor)
        sum_loss_train = pass_tensor.cpu().detach().item()
        ave_loss_train = sum_loss_train / world_size

        if local_rank == 0:
            print(
                f"\nGPU{local_rank} report: Epoch {epoch}/{args.epochs} | Train loss = {ave_loss_train:.4f}\n"
            )
        # 3. Validation on GPUs
        val_metrics_avg, val_loss_list = [], []

        for val_time in range(validation_times):
            dist.barrier()
            ys_val, loss_val, metrics_val = make_validation(args, model, val_loader, device, local_rank)
            performance_val_df = pd.concat([
                performance_val_df,
                pd.DataFrame([[epoch, str(val_time)] + list(metrics_val)],
                             columns=["epoch", "rand_val_num"] + metrics_name)
            ])
            val_loss_list.append(loss_val)
            val_metrics_avg.append(sum(metrics_val[:4]) / 4)

        cur_epoch_performance_df = performance_val_df.iloc[-validation_times:]
        AUC_avg, ACC_avg, MCC_avg, F1_avg = cur_epoch_performance_df.roc_auc.mean(
        ), cur_epoch_performance_df.accuracy.mean(
        ), cur_epoch_performance_df.mcc.mean(
        ), cur_epoch_performance_df.f1.mean()

        print(
            f"GPU{local_rank} Validation of Epoch-{epoch}:  AUC_avg = {AUC_avg:.6f}, ACC_avg = {ACC_avg:.6f}, MCC_avg = {MCC_avg:.6f}, F1-avg = {F1_avg:.6f}"
        )

        ave_loss_val = sum(val_loss_list)/len(val_loss_list)
        ep_avg_val = sum(val_metrics_avg)/len(val_metrics_avg)
        val_result = [
            ave_loss_val, AUC_avg, ACC_avg, MCC_avg, F1_avg, ep_avg_val
        ]
        pass_tensors = torch.tensor(val_result).to(device)

        # 4. Valid result synchronization
        dist.barrier()
        dist.all_reduce(pass_tensors)
        sum_val_result = pass_tensors.cpu().detach().tolist()

        assert len(val_result) == len(sum_val_result)
        for i in range(len(val_result)):
            val_result[i] = sum_val_result[i] / world_size

        if local_rank == 0:
            print(
                f"\nGPU{local_rank} report: Epoch {epoch}/{args.epochs} | Ave_val_loss = {val_result[0]:.6f} | Ave_AUC = {val_result[1]:.6f} | Ave_ACC = {val_result[2]:.6f} | Ave_MCC = {val_result[3]:.6f} | Ave_F1 = {val_result[4]:.6f}"
            )

        # 5. Better Validation Performance
        ep_avg_val = val_result[-1]
        if ep_avg_val > valid_best:
            valid_best, ep_best = ep_avg_val, epoch

            # 6. Save model on local_rank0
            if local_rank == 0:
                print(
                    "============================================================"
                )
                print("Better Validation Performance.")
                print(
                    "============================================================"
                )
                print("Model Saving")
                if not os.path.exists(args.model_path):
                    os.makedirs(args.model_path)
                print(
                    f"****Saving model: Best epoch = {ep_best} | Best Valid Metric = {ep_avg_val:.4f}"
                )

                formatted_today = datetime.date.today().strftime("%y%m%d")

                if args.finetune:
                    new_model_name = f"plm_{args.plm}_Finetune_B{args.batch_size}_LR{args.lr}_seq_{args.plm_input}_fold{args.fold}_ep{ep_best}_{formatted_today}.pkl"
                else:
                    new_model_name = f"plm_{args.plm}_WithoutFinetune_B{args.batch_size}_LR{args.lr}_seq_{args.plm_input}_fold{args.fold}_ep{ep_best}_{formatted_today}.pkl"
                print("*****Path saver: ", new_model_name)
                if args.plm == "baseline_mlp":          # 'baseline_mlp' object has no attribute 'module'
                    torch.save(model.eval().state_dict(),
                            args.model_path + new_model_name)
                else:
                    torch.save(model.module.eval().state_dict(),
                            args.model_path + new_model_name)

        if local_rank == 0:
            print("\n")

        dist.barrier()
        time_train += time_train_ep


        # 7. Early stop
        if epoch - ep_best >= args.early_stop:
            print(
                f"\nGPU{local_rank}-EARLY STOP TRIGGERED, Training totally used {time_train:.2f}s"
            )
            break

        
