import random
import re
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from plm_models import TAPE, ProtBert, ESM2, ProtAlBert, AMPLIFY, load_tokenizer
from tape import ProteinBertConfig
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from train import model_train
from data_loader import data_loader, immrep2023_data_loader
from parameters import read_arguments
from baseline_models import baseline_mlp, baseline_rnn


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

    #dataloaders
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

    #tokenizer
    if "baseline" in args.plm:
        # set protbert tokenizer for baseline model
        tokenizer = load_tokenizer('protbert')
    else:
        tokenizer = load_tokenizer(args.plm)

    #model
    #pdb.set_trace()
    if args.finetune:
        print(f"\n**Start finetuning protein language models [{args.plm}]\n")
        if args.plm == "tape":
            config = ProteinBertConfig.from_pretrained('bert-base')
            model = TAPE(head_type=args.head_type, plm_output=args.plm_output)
        elif args.plm == "protbert":
            model = ProtBert(head_type=args.head_type, plm_output=args.plm_output)
        elif "esm2" in args.plm:
            model = ESM2(head_type=args.head_type, plm_output=args.plm_output,
                        esm_size=args.plm.split('-')[-1])
        elif args.plm =="protalbert":
            model = ProtAlBert(head_type=args.head_type, plm_output=args.plm_output)
        elif "AMPLIFY" in args.plm:
            model = AMPLIFY(head_type=args.head_type, plm_output=args.plm_output,
                        amplify_type=args.plm)

        model.to(device)
        # Load checkpoint if it exists
        if args.checkpoint_file:
            print(f"Load model parameters from {args.checkpoint_file}")
            checkpoint = torch.load(args.checkpoint_file)
            pattern = r"_ep(\d+)_"
            match = re.search(pattern, args.checkpoint_file)
            if match:
                ep_best = int(match.group(1))
                start_epoch = ep_best
            model.load_state_dict(checkpoint)
        else:
            start_epoch = 0

        if torch.cuda.device_count() > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True  # pooler layer is unused
            )

        model_train(args, train_sampler, train_loader, val_loader, model, tokenizer, 
                    device, local_rank, world_size, start_epoch)
        print("\nFinished training")

    else:
        print("\n**Train the projection head only, with frozen PLM\n")
        if args.plm == "tape":
            config = ProteinBertConfig.from_pretrained('bert-base')
            model = TAPE(head_type=args.head_type, plm_output=args.plm_output, finetune_plm = False)
        elif args.plm == "protbert":
            model = ProtBert(head_type=args.head_type, plm_output=args.plm_output, finetune_plm = False)
        elif "esm2" in args.plm:
            model = ESM2(head_type=args.head_type, plm_output=args.plm_output, finetune_plm = False,
                        esm_size=args.plm.split('-')[-1])
        elif args.plm =="protalbert":
            model = ProtAlBert(head_type=args.head_type, plm_output=args.plm_output, finetune_plm = False)
        elif "AMPLIFY" in args.plm:
            model = AMPLIFY(head_type=args.head_type, plm_output=args.plm_output, finetune_plm = False,
                        amplify_type=args.plm)
        elif "baseline" in args.plm:
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

            if args.plm == "baseline_mlp":
                model = baseline_mlp(emb_dim=32, seq_max_len=input_seq_max_len)
            elif args.plm == "baseline_rnn":
                model = baseline_rnn(emb_dim=32, hidden_size=100, output_size=2, rnn_type="lstm")
            else:
                raise ValueError(f"No baseline model named {args.plm}")
            total_params = sum([param.nelement() for param in model.parameters() if param.requires_grad])
            print(">>> Total params: {}".format(total_params))
            print(model)

        model.to(device)
        # Load checkpoint if it exists
        if args.checkpoint_file:
            print(f"Load model parameters from {args.checkpoint_file}")
            checkpoint = torch.load(args.checkpoint_file)
            pattern = r"_ep(\d+)_"
            match = re.search(pattern, args.checkpoint_file)
            if match:
                ep_best = int(match.group(1))
                start_epoch = ep_best
            model.load_state_dict(checkpoint)
        else:
            start_epoch = 0

        if torch.cuda.device_count() > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True  # pooler layer is unused
            )

        model_train(args, train_sampler, train_loader, val_loader, model, tokenizer,
                    device, local_rank, world_size, start_epoch)
        print("\nFinished training")


if __name__ == "__main__":
    main()
