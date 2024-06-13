import os
import random
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.utils.data as Data
from tape import TAPETokenizer
from transformers import AutoTokenizer, AlbertTokenizer


"""
Check your path of pre-trained models
"""
#### from huggingface
# protbert_bfd_checkpoint = "Rostlab/prot_bert_bfd"
# protalbert_checkpoint = "Rostlab/prot_albert"
# esm2_8b_checkpoint = "facebook/esm2_t6_8M_UR50D"
# esm2_35m_checkpoint = "facebook/esm2_t12_35M_UR50D"
# esm2_150m_checkpoint = "facebook/esm2_t30_150M_UR50D"

#### local
protbert_bfd_checkpoint = "/data/lujd/huggingface/hub/prot_bert_bfd"
protalbert_checkpoint = "/data/lujd/huggingface/hub/prot_albert"
esm2_8b_checkpoint = "/data/lujd/huggingface/hub/esm2_t6_8M_UR50D"
esm2_35m_checkpoint = "/data/lujd/huggingface/hub/esm2_t12_35M_UR50D"
esm2_150m_checkpoint = "/data/lujd/huggingface/hub/esm2_t30_150M_UR50D"


class tcr_pmhc_Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 df_tcr_pmhc,
                 pep_max_len,
                 tcr_max_len,
                 padding=True):
        super(tcr_pmhc_Dataset, self).__init__()
        """
        tcr_seq_list: list, size: N
        pep_seq_list: list, size: N
        """
        self.tcr_seq_list = df_tcr_pmhc.tcr.to_list() 
        self.pep_seq_list = df_tcr_pmhc.pep.to_list()  
        self.labels = torch.LongTensor(df_tcr_pmhc.label.to_list())
        self.pep_max_len = pep_max_len
        self.tcr_max_len = tcr_max_len
        self.padding = padding

    def __getitem__(self, index):
        tcr_seq, pep_seq, label = (
            self.tcr_seq_list[index],
            self.pep_seq_list[index],
            self.labels[index]
        )

        if self.padding == True:
            tcr_seq = tcr_seq.ljust(self.tcr_max_len, 'X')
            pep_seq = pep_seq.ljust(self.pep_max_len, 'X')
        '''
        tcr_seq: str
        pep_seq: str
        '''
        return tcr_seq, pep_seq, label

    def __len__(self):
        return len(self.pep_seq_list)


class tcr_pmhc_Dataset_RN(torch.utils.data.Dataset):
    """RN: Random Negative sample generation"""

    def __init__(self,
                 df_tcr_pmhc,
                 neg_tcr_candidate_list, # stores the negative tcr sequences for peptides
                 pep_max_len,
                 tcr_max_len,
                 padding=True):

        super(tcr_pmhc_Dataset_RN, self).__init__()

        self.pep_seq_list = df_tcr_pmhc.pep.to_list()
        self.tcr_seq_list = df_tcr_pmhc.tcr.to_list()  
        self.pep_tcr_mapping = df_tcr_pmhc.groupby('pep')['tcr'].unique().to_dict()

        self.pep_max_len = pep_max_len
        self.tcr_max_len = tcr_max_len
        self.padding = padding
        self.neg_tcr_candidates = neg_tcr_candidate_list

    def __getitem__(self, index):
        pos_tcr_seq, pep_seq = (
            self.tcr_seq_list[index],
            self.pep_seq_list[index],
        )

        binding_true_set = set(self.pep_tcr_mapping.get(pep_seq))
        neg_tcr_seq = random.choice(self.neg_tcr_candidates)
        while neg_tcr_seq in binding_true_set:
            neg_tcr_seq = random.choice(self.neg_tcr_candidates)
        if self.padding == True:
            pos_tcr_seq = pos_tcr_seq.ljust(self.tcr_max_len, 'X')
            pep_seq = pep_seq.ljust(self.pep_max_len, 'X')
            neg_tcr_seq = neg_tcr_seq.ljust(self.tcr_max_len, 'X')

        return pos_tcr_seq, neg_tcr_seq, pep_seq

    def __len__(self):
        return len(self.tcr_seq_list)


class level_2A_Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 df_beta_phla,
                 phla_max_len,
                 tcr_max_len,
                 padding=True):
        super(level_2A_Dataset, self).__init__()
        """
        tcr_seq_list: list, size: N
        pep_seq_list: list, size: N
        hla_seq_list: list, size: N
        """
        self.tcr_seq_list = df_beta_phla.beta.to_list() 
        self.pep_seq_list = df_beta_phla.pep.to_list()  
        # self.hla_seq_list = df_beta_phla.hla.to_list()
        # self.phla_seq_list = [pep + hla for pep, hla in zip(self.pep_seq_list, self.hla_seq_list)]
        if "hla" in df_beta_phla.columns:
            self.hla_seq_list = df_beta_phla.hla.to_list()
            self.phla_seq_list = [pep + hla for pep, hla in zip(self.pep_seq_list, self.hla_seq_list)]
        else:
            self.phla_seq_list = self.pep_seq_list
            print('no hla information')
            print(self.phla_seq_list[:2], self.pep_seq_list[:2])
        self.labels = torch.LongTensor(df_beta_phla.label.to_list())
        self.tcr_max_len = tcr_max_len
        self.phla_max_len = phla_max_len

        self.padding = padding

    def __getitem__(self, index):
        tcr_seq, phla_seq, label = (
            self.tcr_seq_list[index],
            self.phla_seq_list[index],
            self.labels[index]
        )

        if self.padding == True:
            tcr_seq = tcr_seq.ljust(self.tcr_max_len, 'X')
            phla_seq = phla_seq.ljust(self.phla_max_len, 'X')
        '''
        tcr_seq: str
        phla_seq: str
        '''
        return tcr_seq, phla_seq, label

    def __len__(self):
        return len(self.tcr_seq_list)

class level_2A_Dataset_RN(torch.utils.data.Dataset):

    def __init__(self,
                 df_beta_phla,
                 neg_beta_candidate_list,
                 phla_max_len,
                 tcr_max_len,
                 padding=True):
        super(level_2A_Dataset_RN, self).__init__()

        self.tcr_seq_list = df_beta_phla.beta.to_list() 
        self.pep_seq_list = df_beta_phla.pep.to_list()  
        self.pep_tcr_mapping = df_beta_phla.groupby('pep')['beta'].unique().to_dict()

        if "hla" in df_beta_phla.columns:
            self.hla_seq_list = df_beta_phla.hla.to_list()
            self.phla_seq_list = [pep + hla for pep, hla in zip(self.pep_seq_list, self.hla_seq_list)]
        else:
            self.phla_seq_list = self.pep_seq_list
            print('no hla information')
            print(self.phla_seq_list[:2], self.pep_seq_list[:2])

        self.neg_beta_candidates = neg_beta_candidate_list
        self.tcr_max_len = tcr_max_len
        self.phla_max_len = phla_max_len

        self.padding = padding
        # print(self.tcr_max_len, self.phla_max_len)

    def __getitem__(self, index):
        pos_beta_seq, pep_seq, phla_seq = (
            self.tcr_seq_list[index],
            self.pep_seq_list[index],
            self.phla_seq_list[index],
        )

        binding_true_set = set(self.pep_tcr_mapping.get(pep_seq))
        neg_beta_seq = random.choice(self.neg_beta_candidates)
        while neg_beta_seq in binding_true_set:
            neg_beta_seq = random.choice(self.neg_beta_candidates)
        if self.padding == True:
            pos_beta_seq = pos_beta_seq.ljust(self.tcr_max_len, 'X')
            phla_seq = phla_seq.ljust(self.phla_max_len, 'X')
            neg_beta_seq = neg_beta_seq.ljust(self.tcr_max_len, 'X')

        return pos_beta_seq, neg_beta_seq, phla_seq

    def __len__(self):
        return len(self.tcr_seq_list)

# class level_2B_Dataset_RN(torch.utils.data.Dataset):

#     def __init__(self,
#                  df_pep_ab,
#                  neg_tcr_candidate_list, 
#                  pep_max_len,
#                  ab_max_len,
#                  padding=True):
#         super(level_2B_Dataset_RN, self).__init__()

#         self.tcr_seq_list = df_pep_ab.ab.to_list() 
#         self.pep_seq_list = df_pep_ab.pep.to_list()  

#         self.pep_tcr_mapping = df_pep_ab.groupby('pep')['ab'].unique().to_dict()

#         self.neg_tcr_candidates = neg_tcr_candidate_list
#         self.ab_max_len = ab_max_len
#         self.pep_max_len = pep_max_len

#         self.padding = padding

#     def __getitem__(self, index):
#         pos_tcr_seq, pep_seq = (
#             self.tcr_seq_list[index],
#             self.pep_seq_list[index],
#         )

#         binding_true_set = set(self.pep_tcr_mapping.get(pep_seq))
#         neg_tcr_seq = random.choice(self.neg_tcr_candidates)
#         while neg_tcr_seq in binding_true_set:
#             neg_tcr_seq = random.choice(self.neg_tcr_candidates)
#         if self.padding == True:
#             pos_tcr_seq = pos_tcr_seq.ljust(self.ab_max_len, 'X')
#             pep_seq = pep_seq.ljust(self.pep_max_len, 'X')
#             neg_tcr_seq = neg_tcr_seq.ljust(self.ab_max_len, 'X')

#         return pos_tcr_seq, neg_tcr_seq, pep_seq

#     def __len__(self):
#         return len(self.tcr_seq_list)


class level_2B_Dataset_RN(torch.utils.data.Dataset):

    def __init__(self,
                 df_pep_ab,
                 neg_tcr_candidate_list, 
                 pep_max_len,
                 ab_max_len,
                 padding=True):
        super(level_2B_Dataset_RN, self).__init__()

        self.pep_seq_list = df_pep_ab.pep.to_list()

        if "ab" in df_pep_ab.columns:
            self.tcr_seq_list = df_pep_ab.ab.to_list() 
            self.pep_tcr_mapping = df_pep_ab.groupby('pep')['ab'].unique().to_dict()
        else:
            self.tcr_seq_list = df_pep_ab.beta.to_list() 
            self.pep_tcr_mapping = df_pep_ab.groupby('pep')['beta'].unique().to_dict()
            print('no alpha information')
            print(self.tcr_seq_list[:2])

        self.neg_tcr_candidates = neg_tcr_candidate_list
        self.ab_max_len = ab_max_len
        self.pep_max_len = pep_max_len

        self.padding = padding
        self.print_tag = 1

    def __getitem__(self, index):
        pos_tcr_seq, pep_seq = (
            self.tcr_seq_list[index],
            self.pep_seq_list[index],
        )

        if self.print_tag == 1:
            print(pep_seq, pos_tcr_seq)

        binding_true_set = set(self.pep_tcr_mapping.get(pep_seq))
        neg_tcr_seq = random.choice(self.neg_tcr_candidates)
        while neg_tcr_seq in binding_true_set:
            neg_tcr_seq = random.choice(self.neg_tcr_candidates)
        if self.padding == True:
            pep_seq = pep_seq.ljust(self.pep_max_len, 'X')
            pos_tcr_seq = ''.join([s.ljust(self.ab_max_len, 'X') for s in pos_tcr_seq.split('/')])
            neg_tcr_seq = ''.join([s.ljust(self.ab_max_len, 'X') for s in neg_tcr_seq.split('/')])

        if self.print_tag == 1:     # print once
            print(pep_seq, pos_tcr_seq, neg_tcr_seq)
            self.print_tag = 0

        return pos_tcr_seq, neg_tcr_seq, pep_seq

    def __len__(self):
        return len(self.tcr_seq_list)


class level_3_Dataset_RN(torch.utils.data.Dataset):

    def __init__(self,
                 df_phla_ab,
                 neg_tcr_candidate_list, 
                 pep_max_len,
                 ab_max_len,
                 padding=True,
                 hla_max_len=34):
        super(level_3_Dataset_RN, self).__init__()

        self.pep_seq_list = df_phla_ab.pep.to_list()  

        if "ab" in df_phla_ab.columns:
            self.tcr_seq_list = df_phla_ab.ab.to_list() 
            self.pep_tcr_mapping = df_phla_ab.groupby('pep')['ab'].unique().to_dict()
        else:
            self.tcr_seq_list = df_phla_ab.beta.to_list() 
            self.pep_tcr_mapping = df_phla_ab.groupby('pep')['beta'].unique().to_dict()
            print('no alpha information')
            print(self.tcr_seq_list[:2])
        
        if "hla" in df_phla_ab.columns:
            self.hla_seq_list = df_phla_ab.hla.to_list()
            self.phla_seq_list = [pep+'/'+hla for pep, hla in zip(self.pep_seq_list, self.hla_seq_list)]
        else:
            self.phla_seq_list = self.pep_seq_list
            print('no hla information')
            print(self.phla_seq_list[:2], self.pep_seq_list[:2])

        self.neg_tcr_candidates = neg_tcr_candidate_list
        self.ab_max_len = ab_max_len
        self.pep_max_len = pep_max_len
        self.hla_max_len = hla_max_len

        self.padding = padding
        self.print_tag = 0

    def __getitem__(self, index):
        pos_tcr_seq, pep_seq, phla_seq = (
            self.tcr_seq_list[index],
            self.pep_seq_list[index],
            self.phla_seq_list[index]
        )

        if self.print_tag == 1:
            print(phla_seq, pos_tcr_seq)

        binding_true_set = set(self.pep_tcr_mapping.get(pep_seq))
        neg_tcr_seq = random.choice(self.neg_tcr_candidates)
        while neg_tcr_seq in binding_true_set:
            neg_tcr_seq = random.choice(self.neg_tcr_candidates)
        if self.padding == True:                        # pad then cat
            if len(phla_seq.split('/')) > 1:
                phla_seq = pep_seq.ljust(self.pep_max_len, 'X') + phla_seq.split('/')[-1].ljust(self.hla_max_len, 'X')
            else:
                phla_seq = pep_seq.ljust(self.pep_max_len, 'X')
            pos_tcr_seq = ''.join([s.ljust(self.ab_max_len, 'X') for s in pos_tcr_seq.split('/')])
            neg_tcr_seq = ''.join([s.ljust(self.ab_max_len, 'X') for s in neg_tcr_seq.split('/')])

        ### cat then pad for tcr
        # pos_tcr_seq = ''.join(pos_tcr_seq.split('/')).ljust(self.ab_max_len*len(pos_tcr_seq.split('/')), 'X')
        # neg_tcr_seq = ''.join(neg_tcr_seq.split('/')).ljust(self.ab_max_len*len(neg_tcr_seq.split('/')), 'X')

        if self.print_tag == 1:     # print once
            print(phla_seq, pos_tcr_seq, neg_tcr_seq)
            self.print_tag = 0
        return pos_tcr_seq, neg_tcr_seq, phla_seq

    def __len__(self):
        return len(self.tcr_seq_list)


def seq2token(tcr_seq_list, pep_seq_list, plm_type, plm_input_type, device, esm_size = '8M'):
    '''
    plm_type : "tape" "protbert/protalbert" "esm"
    plm_input_type: "sep", "cat"
    '''

    tcr_pep_inputs = []  # the input of model is token

    if plm_type == "tape":
        tokenizer = TAPETokenizer(vocab='iupac')
        for tcr, pep in zip(tcr_seq_list, pep_seq_list):
            tcr_pmhc = tcr + pep
            token = tokenizer.encode(tcr_pmhc)  # array
            if plm_input_type == "sep":
                token = np.insert(token, (len(tcr_pmhc) + 1),
                                  3)  # insert 3(<sep>) in position len(hla)+1
            tcr_pep_inputs.append(token)

    elif plm_type in ['protbert', 'protalbert']:
        if plm_type == 'protbert':
            tokenizer = AutoTokenizer.from_pretrained(protbert_bfd_checkpoint,
                                                    do_lower_case=False)
        elif plm_type == 'protalbert':
            tokenizer = AlbertTokenizer.from_pretrained(protalbert_checkpoint,
                                                    do_lower_case=False)
        for tcr, pep in zip(tcr_seq_list, pep_seq_list):
            tcr_pmhc = tcr + pep
            tcr_pmhc = ' '.join(tcr_pmhc)
            token = tokenizer.encode(tcr_pmhc)  # array
            if plm_input_type == "sep":
                token = np.insert(token, (len(tcr_pmhc) + 1),
                                  3)  # insert 3(<sep>) in position len(hla)+1
            tcr_pep_inputs.append(token)

    elif "esm" in plm_type:
        if plm_type == 'esm':                       # default: 8M
            tokenizer = AutoTokenizer.from_pretrained(esm2_8b_checkpoint)
        elif plm_type.split('-')[-1] == '35M':      # actually same as 8M
            tokenizer = AutoTokenizer.from_pretrained(esm2_35m_checkpoint)
        elif plm_type.split('-')[-1] == '150M':     # actually same as 8M
            tokenizer = AutoTokenizer.from_pretrained(esm2_150m_checkpoint)
        for tcr, pep in zip(tcr_seq_list, pep_seq_list):
            tcr_pmhc = tcr + pep
            token = tokenizer.encode(tcr_pmhc)  # array
            if plm_input_type == "sep":
                token = np.insert(token, (len(tcr_pmhc) + 1),
                                  3)  # insert 3(<sep>) in position len(hla)+1
            tcr_pep_inputs.append(token)

    tcr_pep_inputs = np.array(tcr_pep_inputs)
    tcr_pep_inputs_tensor = torch.from_numpy(tcr_pep_inputs)
    tcr_pep_inputs_tensor = tcr_pep_inputs_tensor.to(device)
    return tcr_pep_inputs_tensor


# Dataloder seed
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def data_loader(data_path, batch_size, fold, rand_neg, num_workers, pep_max_len, tcr_max_len, level):
    train_df = pd.read_csv(os.path.join(data_path,"train_data_fold{}.csv".format(fold)))
    valid_df = pd.read_csv(os.path.join(data_path,"valid_data_fold{}.csv".format(fold)))
    test_df = pd.read_csv(os.path.join(data_path,"test_data_fold{}.csv".format(fold)))

    tcr2candidates = np.load(os.path.join(data_path, "tcr2candidates_pools.npy"), allow_pickle=True,).tolist()  ##ranking???

    if "1" in level:
        if rand_neg:
            train_pos_df = train_df[train_df.label == 1]  #select the positive samples
            valid_pos_df = valid_df[valid_df.label == 1]
            test_pos_df = test_df[test_df.label == 1]
            val_dataset = tcr_pmhc_Dataset_RN(valid_pos_df, tcr2candidates, pep_max_len, tcr_max_len)
            train_dataset = tcr_pmhc_Dataset_RN(train_pos_df, tcr2candidates, pep_max_len, tcr_max_len)
            test_dataset = tcr_pmhc_Dataset_RN(test_pos_df, tcr2candidates, pep_max_len, tcr_max_len)
        else:
            train_dataset = tcr_pmhc_Dataset(train_df, pep_max_len, tcr_max_len)
            val_dataset = tcr_pmhc_Dataset(valid_df, pep_max_len, tcr_max_len)
            test_dataset = tcr_pmhc_Dataset(test_df, pep_max_len, tcr_max_len)
    elif "2a" in level:
        if rand_neg:
            train_pos_df = train_df[train_df.label == 1]  #select the positive samples
            valid_pos_df = valid_df[valid_df.label == 1]
            test_pos_df = test_df[test_df.label == 1]
            val_dataset = level_2A_Dataset_RN(valid_pos_df, tcr2candidates, pep_max_len, tcr_max_len)
            train_dataset = level_2A_Dataset_RN(train_pos_df, tcr2candidates, pep_max_len, tcr_max_len)
            test_dataset = level_2A_Dataset_RN(test_pos_df, tcr2candidates, pep_max_len, tcr_max_len)
        else:
            train_dataset = level_2A_Dataset(train_df, pep_max_len, tcr_max_len)
            val_dataset = level_2A_Dataset(valid_df, pep_max_len, tcr_max_len)
            test_dataset = level_2A_Dataset(test_df, pep_max_len, tcr_max_len)
    elif "2b" in level:
        if rand_neg:
            train_pos_df = train_df[train_df.label == 1]  #select the positive samples
            valid_pos_df = valid_df[valid_df.label == 1]
            test_pos_df = test_df[test_df.label == 1]
            val_dataset = level_2B_Dataset_RN(valid_pos_df, tcr2candidates, pep_max_len, tcr_max_len)
            train_dataset = level_2B_Dataset_RN(train_pos_df, tcr2candidates, pep_max_len, tcr_max_len)
            test_dataset = level_2B_Dataset_RN(test_pos_df, tcr2candidates, pep_max_len, tcr_max_len)
    elif ("3" in level) or ("4" in level):
        if rand_neg:
            train_pos_df = train_df[train_df.label == 1]  #select the positive samples
            valid_pos_df = valid_df[valid_df.label == 1]
            test_pos_df = test_df[test_df.label == 1]
            val_dataset = level_3_Dataset_RN(valid_pos_df, tcr2candidates, pep_max_len, tcr_max_len)
            train_dataset = level_3_Dataset_RN(train_pos_df, tcr2candidates, pep_max_len, tcr_max_len)
            test_dataset = level_3_Dataset_RN(test_pos_df, tcr2candidates, pep_max_len, tcr_max_len)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    train_loader = Data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    val_loader = Data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    test_loader = Data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    return train_loader, train_sampler, val_loader, test_loader




'''
seed = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
# Dataloder seed
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    # init GPU process
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()  # GPU_rank
    world_size = dist.get_world_size()  # GPU_num
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    data_path = "/data/luyq/tcr-pmhc"
    df_tcr_pmhc = pd.read_csv(os.path.join(data_path,"df_tcr_pmhc.csv"),index_col=0)
    dataset_tcr_pmhc = tcr_pmhc_Dataset(df_tcr_pmhc, pep_max_len=15, tcr_max_len=15)
    train_sampler = DistributedSampler(dataset_tcr_pmhc,shuffle=True)
    train_loader = Data.DataLoader(
        dataset_tcr_pmhc,
        batch_size=14,
        shuffle=False,
        sampler=train_sampler,
        num_workers=0,
        worker_init_fn=seed_worker,
    )
    for data in train_loader:
        tcr, pmhc = data
        print(tcr)
        print("\n")
        print(pmhc)


if __name__ == "__main__":
    main()'''