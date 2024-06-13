import argparse

def read_arguments():
    parser = argparse.ArgumentParser(description="""Program entry point for tcr_pmhc binding prediction training""")

    # basic configuration
    parser.add_argument("--data_path",
                        type=str,
                        default="/data/luyq/level2b")
    # The folder containing tcr+pmhc pairs and the corresponding sequences, etc.
    
    parser.add_argument("--plm_path",
                        type=str,
                        default="/data/luyq/")
    # The folder containing pretrained PLM checkpoints, which is still empty currently.
    parser.add_argument("--model_path",
                        type=str,
                        default="/data/luyq/")
    parser.add_argument("--tcr_max_len", type=int, default=34)
    parser.add_argument("--pep_max_len", type=int, default=9)
    parser.add_argument("--threshold", type=float, default=0.5)

    # -----------------Parameters for Datasets------------------------
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--level", type=str, choices=["1", 
                                                      "2a", "2a_basic", 
                                                      "2b", "2b_basic", 
                                                      "3", "3_basic", 
                                                      "4", "4_basic",
                                                      "4_one","4_twoa","4_twob","4_three","4_four"],
                        default="1")

    # Do not change the fold


    # -----------------Parameters for Model Architectures-----------------------
    parser.add_argument("--plm",
                        type=str,
                        choices=["tape", "protbert", "esm", "esm-150M", "esm-35M",
                                 "protalbert", "baseline_mlp", "baseline_rnn"],
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
    parser.add_argument("--rand_neg", type=bool, default=True)
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--checkpoint_file", type=str, default=None)

    parser.add_argument("--eval_setting",
                        type=str,
                        choices=["classification", "ranking"],
                        default="classification")

    # -----------------Parameters for Distributed------------------------
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    return args

