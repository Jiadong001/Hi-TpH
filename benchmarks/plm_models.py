import numpy as np
import torch
import torch.nn as nn
from projection_head import MLP
from tape import ProteinBertAbstractModel, ProteinBertModel, TAPETokenizer
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


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


'''
TAPE model
'''


class TAPE(nn.Module):

    def __init__(self, head_type='3MLP', plm_output='mean', finetune_plm = True):
        super(TAPE, self).__init__()
        self.tape = ProteinBertModel.from_pretrained('bert-base')
        self.head_type = head_type
        self.plm_output = plm_output
        self.finetune_plm = finetune_plm
        # Freeze the parameters of the PLM if finetune_plm is False
        if not finetune_plm:
            for param in self.tape.parameters():
                param.requires_grad = False

        if head_type == '3MLP':
            self.projection = MLP(768, [256, 64, 2])  ## 3layers
        elif head_type == '5MLP':
            self.projection = MLP(768, [1024, 512, 128, 32, 2])  ## 5layers

    def forward(self, inputs):
        if self.plm_output == 'mean':  # mean of sequence_output
            outputs = self.tape(inputs)[
                0]  # [batch_size, , seq_len,  hidden_size]
            outputs = torch.mean(outputs, dim=1)  # [batch_size, hidden_size]
            outputs = self.projection(outputs)
        elif self.plm_output == 'cls':  # output of every sample's <cls> token
            outputs = self.tape(
                inputs, input_mask=None)[0][:, 0]  # [batch_size, hidden_size]
            outputs = self.projection(outputs)
        return outputs.view(-1, outputs.size(-1))


'''
ProteinBert model
'''


class ProtBert(nn.Module):

    def __init__(self, head_type='3MLP', plm_output='mean', finetune_plm = True):
        super(ProtBert, self).__init__()
        #self.proteinbert = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")
        self.proteinbert = AutoModel.from_pretrained(protbert_bfd_checkpoint)

        self.head_type = head_type
        self.plm_output = plm_output
        self.finetune_plm = finetune_plm

        # Freeze the parameters of the PLM if finetune_plm is False
        if not finetune_plm:
            for param in self.proteinbert.parameters():
                param.requires_grad = False

        if head_type == '3MLP':
            self.projection = MLP(1024, [256, 64, 2])  ## 3layers
        elif head_type == '5MLP':
            self.projection = MLP(1024, [1024, 512, 128, 32, 2])  ## 5layers
        self.plm_output = plm_output

    def forward(self, input_ids):
        outputs = self.proteinbert(input_ids)

        if self.plm_output == 'mean':
            outputs = outputs[0].mean(dim=1)  # [batch_size, hidden_size]
            outputs = self.projection(outputs)
        elif self.plm_output == 'cls':
            outputs = outputs[0][:, 0]
            outputs = self.projection(outputs)

        return outputs.view(-1, outputs.size(-1))


'''
ProtAlBert model
'''


class ProtAlBert(nn.Module):

    def __init__(self, head_type='3MLP', plm_output='mean', finetune_plm=True):
        super(ProtAlBert, self).__init__()
        self.proteinbert = AutoModel.from_pretrained(protalbert_checkpoint)
        self.head_type = head_type
        self.plm_output = plm_output
        self.finetune_plm = finetune_plm

        # Freeze the parameters of the PLM if finetune_plm is False
        if not finetune_plm:
            for param in self.proteinbert.parameters():
                param.requires_grad = False

        if head_type == '3MLP':
            self.projection = MLP(4096, [256, 64, 2])               ## 3layers
        elif head_type == '5MLP':
            self.projection = MLP(4096, [1024, 512, 128, 32, 2])    ## 5layers
        self.plm_output = plm_output

    def forward(self, input_ids):
        outputs = self.proteinbert(input_ids)

        if self.plm_output == 'mean':
            outputs = outputs[0].mean(dim=1)  # [batch_size, hidden_size]
            outputs = self.projection(outputs)
        elif self.plm_output == 'cls':
            outputs = outputs[0][:, 0]
            outputs = self.projection(outputs)

        return outputs.view(-1, outputs.size(-1))


'''
ESM family
'''
class ESM(nn.Module): 
    def __init__(self, head_type='3MLP', plm_output='mean', finetune_plm = True, esm_size = '8M'):
        super(ESM, self).__init__()
        if esm_size == '8M':
            self.checkpoint = esm2_8b_checkpoint
            self.hidden_size = 320
        elif esm_size == '35M':
            self.checkpoint = esm2_35m_checkpoint
            self.hidden_size = 480
        elif esm_size == '150M':
            self.checkpoint = esm2_150m_checkpoint
            self.hidden_size = 640
        # elif esm_size == '650M':
        #     self.checkpoint = "facebook/esm2_t33_650M_UR50D"
        # elif esm_size == '3B':
        #     self.checkpoint = "facebook/esm2_t36_3B_UR50D"
        # elif esm_size == '15B':
        #     self.checkpoint = "facebook/esm2_t48_15B_UR50D"
        else:
            raise ValueError(f"Wrong size of ESM: {esm_size}")
        self.esm = AutoModel.from_pretrained(self.checkpoint)
        self.head_type = head_type
        self.plm_output = plm_output
        self.finetune_plm = finetune_plm
        print(self.plm_output)
        # print(self.hidden_size)

        # Freeze the parameters of the PLM if finetune_plm is False
        if not finetune_plm:
            for param in self.esm.parameters():
                param.requires_grad = False
        
        if head_type == '3MLP':
            self.projection = MLP(self.hidden_size, [256, 64, 2])  ## 3layers
        elif head_type == '5MLP':
            self.projection = MLP(self.hidden_size, [1024, 512, 128, 32, 2])  ## 5layers
        self.plm_output = plm_output

    def forward(self, input_ids):
        outputs = self.esm(input_ids)
        if self.plm_output == 'mean':
            outputs = outputs[0].mean(dim=1)  # [batch_size, hidden_size]
            outputs = self.projection(outputs)
        elif self.plm_output == 'cls':
            outputs = outputs[0][:, 0]
            outputs = self.projection(outputs)

        return outputs.view(-1, outputs.size(-1))

