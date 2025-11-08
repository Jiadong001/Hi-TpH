import argparse

def read_arguments():
    parser = argparse.ArgumentParser(description="""Program entry point for tcr_pmhc binding prediction training""")

    # basic configuration
    parser.add_argument("--data_path",
                        type=str,
                        default="../../benchmarks_dataset/level1")
    # The folder containing tcr+pmhc pairs and the corresponding sequences, etc.

    # The folder containing pretrained PLM checkpoints, which is still empty currently.
    parser.add_argument("--model_path",
                        type=str,
                        default="/data/luyq/")
    parser.add_argument("--tcr_max_len", type=int, default=34)
    parser.add_argument("--pep_max_len", type=int, default=9)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    # -----------------Parameters for Datasets------------------------
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--level", type=str, choices=["1", 
                                                      "2", "2_basic", 
                                                      "3", "3_basic", 
                                                      "4", "4_basic",
                                                      ],
                        default="1")
    # parser.add_argument('--components', nargs='+', type=str, default=['pep', 'beta'])
    parser.add_argument('--components', type=str, 
                        choices=['pep beta', 'pep ab', 'pep hla beta', 'pep hla ab',
                                 'pep beta_cdr3', 'pep hla beta_cdr3', 'pep hla ab_cdr3'],
                        default='pep beta')
    # Do not change the fold

    # -----------------Parameters for Model Architectures-----------------------
    parser.add_argument("--plm",
                        type=str,
                        choices=["tape", "protbert", "protalbert", 
                            "esm2-8M", "esm2-35M", "esm2-150M", "esm2-650M", "esm2-3B",
                            "AMPLIFY-120M", "AMPLIFY-120M-base", 'AMPLIFY-350M', 'AMPLIFY-350M-base',
                            "baseline_mlp", "baseline_rnn"],
                        default="tape")
    parser.add_argument("--plm_input",
                        type=str,
                        choices=["cat", "sep"],
                        default="cat")
    parser.add_argument("--plm_output",
                        type=str,
                        choices=["mean", "cls"],
                        default="cls")
    parser.add_argument("--head_type", type=str, default="3MLP")

    # -----------------Parameters for training------------------------
    parser.add_argument("--finetune",
                        default=False,
                        action="store_true",
                        help="whether to update parameters of PLM")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--rand_neg", default=False, action="store_true")
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--checkpoint_file", type=str, default=None)

    # -----------------Parameters for testing-------------------------
    parser.add_argument("--test_data",
                        type=str,
                        choices=["ours", "immrep2023"],
                        default="ours")

    # -----------------Parameters for Distributed------------------------
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    args.components = args.components.split() #!

    return args

