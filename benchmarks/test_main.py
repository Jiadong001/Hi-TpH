import os
import random
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from plm_models import TAPE, ProtBert, ESM2, ProtAlBert
from tape import ProteinBertConfig
from torch.nn.parallel import DistributedDataParallel
from train import make_validation
from data_loader import data_loader, immrep2023_data_loader
from parameters import read_arguments
from baseline_models import baseline_mlp, baseline_rnn
import datetime

def set_seed(seed):
    # Python & Numpy seed
    random.seed(seed)
    np.random.seed(seed)
    # PyTorch seed
    torch.manual_seed(seed)     # default generator
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CUDNN seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    args = read_arguments()
    print(args)

    data_path = args.data_path
    set_seed(args.seed)

    # init GPU process
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()  # GPU_rank
    world_size = dist.get_world_size()  # GPU_num
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # dataloaders
    if args.test_data == 'ours':
        train_loader, train_sampler, val_loader, test_loader, external_loader = data_loader(
            data_path,
            batch_size=args.batch_size,
            fold=args.fold,
            rand_neg=args.rand_neg,
            tcr_max_len=args.tcr_max_len,
            pep_max_len= args.pep_max_len,
            num_workers=0,
            level=args.level,
            comp_cols=args.components)
        print('Use our data')
    elif args.test_data == 'immrep2023':
        train_loader, train_sampler, val_loader, test_loader, external_loader = immrep2023_data_loader(
                    data_path,
                    batch_size=args.batch_size,
                    fold=args.fold,
                    rand_neg=args.rand_neg,
                    tcr_max_len=args.tcr_max_len,
                    pep_max_len= args.pep_max_len,
                    num_workers=0,
                    level=args.level,
                    comp_cols=args.components)
        print('Use IMMREP2023 data')

    if local_rank == 0:
        print("Number of samples in train_loader: ", len(train_loader.dataset))
        print("Number of samples in val_loader: ", len(val_loader.dataset))
        print("Number of samples in test_loader: ", len(test_loader.dataset))
        if external_loader != None:
            print("Number of samples in external_loader: ", len(external_loader.dataset))

    # load model
    if "WithoutFinetune" in args.checkpoint_file.split('/')[-1]:
        finetune_plm_tag = False
    else:
        finetune_plm_tag = True
    
    if "tape" in args.checkpoint_file.split('/')[-1]:
        config = ProteinBertConfig.from_pretrained('bert-base')
        model = TAPE(head_type=args.head_type, plm_output=args.plm_output, finetune_plm=finetune_plm_tag)
        args.plm = "tape"
    elif "protbert" in args.checkpoint_file.split('/')[-1]:
        model = ProtBert(head_type=args.head_type, plm_output=args.plm_output, finetune_plm=finetune_plm_tag)
        args.plm = "protbert"
    elif "esm2" in args.checkpoint_file.split('/')[-1]:
        esm_sizes = ['8M', '35M', '150M', '650M']
        for esm_size in esm_sizes:
            if f"esm2-{esm_size}" in args.checkpoint_file.split('/')[-1]:
                model = ESM2(head_type=args.head_type, plm_output=args.plm_output, finetune_plm=finetune_plm_tag,
                        esm_size=esm_size)
                args.plm = f"esm2-{esm_size}"
                break
    elif "protalbert" in args.checkpoint_file.split('/')[-1]:
        model = ProtAlBert(head_type=args.head_type, plm_output=args.plm_output, finetune_plm=finetune_plm_tag)
        args.plm = "protalbert"
    elif "baseline" in args.checkpoint_file.split('/')[-1]:
        if ("3" == args.level) or ("4" == args.level):
            hla_max_length = 34
            input_seq_max_len = (args.tcr_max_len*2+args.pep_max_len+2+hla_max_length 
                                if args.plm_input == "cat" 
                                else args.tcr_max_len+args.pep_max_len+3+hla_max_length)
        else:
            input_seq_max_len = (args.tcr_max_len+args.pep_max_len+2 
                            if args.plm_input == "cat" 
                            else args.tcr_max_len+args.pep_max_len+3)
        print(input_seq_max_len)

        if "baseline_mlp" in args.checkpoint_file.split('/')[-1]:
            model = baseline_mlp(emb_dim=32, seq_max_len=input_seq_max_len)
            args.plm = "baseline_mlp"
        elif "baseline_rnn" in args.checkpoint_file.split('/')[-1]:
            model = baseline_rnn(emb_dim=32, hidden_size=100, output_size=2, rnn_type="lstm")
            args.plm = "baseline_rnn"
        print(model)
    else:
        raise ValueError(f"{args.checkpoint_file.split('/')[-1]} does not belong to [tape, protbert/protalbert, esm, baseline_mlp/rnn]!")

    print(f"Checkpoint[{args.checkpoint_file.split('/')[-1]}] is a {args.plm}")
    model.to(device)

    # Load checkpoint if it exists
    checkpoint_path = os.path.join(args.model_path, args.checkpoint_file.split('/')[-1])
    if os.path.exists(checkpoint_path):
        print(f"Load model parameters from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"{checkpoint_path} does not exist!")

    if torch.cuda.device_count() > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True  # pooler layer is unused
        )

    #*******************testing************************   
    # Test the best epoch (with best valid metrics)
    print("\n**Testing\n")
    metrics_name = [
        "roc_auc", "accuracy", "mcc", "f1", "sensitivity", "specificity",
        "precision", "recall", "aupr"
        ]

    if not args.rand_neg:
        test_times = 1
    else:
        test_times = 5
    print(f"test_time: {test_times}")
    
    # Test set
    print("\n==========Test set==========\n")
    test_metrics_avg, test_loss_list = [], []
    for test_time in range(test_times):
        dist.barrier()
        ys_test, loss_test, metrics_test = make_validation(args, model, test_loader, device, local_rank)
        if test_time == 0:
            performance_test_df = pd.DataFrame([list(metrics_test)],
                                            columns=metrics_name)
        else:
            performance_test_df = pd.concat([
                    performance_test_df,
                    pd.DataFrame([list(metrics_test)],
                                columns=metrics_name)
                ])
        # print(performance_test_df)
        test_loss_list.append(loss_test)
        test_metrics_avg.append(sum(metrics_test[:4]) / 4)

    cur_epoch_performance_df = performance_test_df.iloc[-5:]
    AUC_avg, ACC_avg, MCC_avg, F1_avg = cur_epoch_performance_df.roc_auc.mean(
        ), cur_epoch_performance_df.accuracy.mean(
        ), cur_epoch_performance_df.mcc.mean(
        ), cur_epoch_performance_df.f1.mean()

    print(
            f"GPU{local_rank} test :  AUC_avg = {AUC_avg:.6f}, ACC_avg = {ACC_avg:.6f}, MCC_avg = {MCC_avg:.6f}, F1-avg = {F1_avg:.6f}"
        )

    ave_loss_val = sum(test_loss_list)/len(test_loss_list)
    ep_avg_val = sum(test_metrics_avg)/len(test_metrics_avg)
    test_result = [
            ave_loss_val, AUC_avg, ACC_avg, MCC_avg, F1_avg, ep_avg_val
        ]

    pass_tensors = torch.tensor(test_result).to(device)

    dist.barrier()
    dist.all_reduce(pass_tensors)
    sum_test_result = pass_tensors.cpu().detach().tolist()

    assert len(test_result) == len(sum_test_result)
    for i in range(len(test_result)):
            test_result[i] = sum_test_result[i] / world_size

    if local_rank == 0:
        print(
                f"\nGPU{local_rank} test results report:  | Ave_test_loss = {test_result[0]:.6f} | Ave_AUC = {test_result[1]:.6f} | \
                Ave_ACC = {test_result[2]:.6f} | Ave_MCC = {test_result[3]:.6f} | Ave_F1 = {test_result[4]:.6f} " )
    
    
    # External set
    if external_loader != None:
        print("\n==========External set==========\n")
        test_metrics_avg, test_loss_list = [], []
        for test_time in range(test_times):
            dist.barrier()
            ys_test, loss_test, metrics_test = make_validation(args, model, external_loader, device, local_rank)
            if test_time == 0:
                performance_test_df = pd.DataFrame([list(metrics_test)],
                                                columns=metrics_name)
            else:
                performance_test_df = pd.concat([
                        performance_test_df,
                        pd.DataFrame([list(metrics_test)],
                                    columns=metrics_name)
                    ])
            # print(performance_test_df)
            test_loss_list.append(loss_test)
            test_metrics_avg.append(sum(metrics_test[:4]) / 4)

        cur_epoch_performance_df = performance_test_df.iloc[-5:]
        AUC_avg, ACC_avg, MCC_avg, F1_avg = cur_epoch_performance_df.roc_auc.mean(
            ), cur_epoch_performance_df.accuracy.mean(
            ), cur_epoch_performance_df.mcc.mean(
            ), cur_epoch_performance_df.f1.mean()

        print(
                f"GPU{local_rank} external:  AUC_avg = {AUC_avg:.6f}, ACC_avg = {ACC_avg:.6f}, MCC_avg = {MCC_avg:.6f}, F1-avg = {F1_avg:.6f}"
            )

        ave_loss_val = sum(test_loss_list)/len(test_loss_list)
        ep_avg_val = sum(test_metrics_avg)/len(test_metrics_avg)
        test_result = [
                ave_loss_val, AUC_avg, ACC_avg, MCC_avg, F1_avg, ep_avg_val
            ]

        pass_tensors = torch.tensor(test_result).to(device)

        dist.barrier()
        dist.all_reduce(pass_tensors)
        sum_test_result = pass_tensors.cpu().detach().tolist()

        assert len(test_result) == len(sum_test_result)
        for i in range(len(test_result)):
                test_result[i] = sum_test_result[i] / world_size

        if local_rank == 0:
            print(
                    f"\nGPU{local_rank} external results report:  | Ave_test_loss = {test_result[0]:.6f} | Ave_AUC = {test_result[1]:.6f} | \
                    Ave_ACC = {test_result[2]:.6f} | Ave_MCC = {test_result[3]:.6f} | Ave_F1 = {test_result[4]:.6f} " )



if __name__ == "__main__":
    print("\n"+"="*30)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("="*30)
    main()
    print("\nEnd "+"="*30+"\n\n")
