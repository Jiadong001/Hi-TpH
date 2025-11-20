import torch
import torch.nn as nn
from baseline_models import MLP
from tape import ProteinBertModel, TAPETokenizer
from transformers import AutoModel, AutoTokenizer, AlbertTokenizer


"""
Check your path of pre-trained models
"""
#### from huggingface directly
cache_dir = "/NAS/lujd/huggingface/hub/"
protbert_checkpoint = "Rostlab/prot_bert"   # "Rostlab/prot_bert_bfd"
amplify_120m_checkpoint = "chandar-lab/AMPLIFY_120M"
amplify_120m_base_checkpoint = "chandar-lab/AMPLIFY_120M_base"
amplify_350m_checkpoint = "chandar-lab/AMPLIFY_350M"
esm2_650m_checkpoint = "facebook/esm2_t33_650M_UR50D"

#### local (downloaded from huggingface website)
protalbert_checkpoint = "/data/lujd/huggingface/hub/prot_albert"
esm2_8m_checkpoint = "/data/lujd/huggingface/hub/esm2_t6_8M_UR50D"
esm2_35m_checkpoint = "/data/lujd/huggingface/hub/esm2_t12_35M_UR50D"
esm2_150m_checkpoint = "/data/lujd/huggingface/hub/esm2_t30_150M_UR50D"
esm2_3b_checkpoint = "/NAS/lujd/huggingface/hub/esm2_t36_3B_UR50D"


'''
Tokenizer
'''
def load_tokenizer(plm_type):
    if plm_type == "tape":
        tokenizer = TAPETokenizer(vocab='iupac')
    elif plm_type in ['protbert', 'protalbert']:
        if plm_type == 'protbert':
            tokenizer = AutoTokenizer.from_pretrained(protbert_checkpoint, cache_dir=cache_dir,
                                                    do_lower_case=False)
        elif plm_type == 'protalbert':
            tokenizer = AlbertTokenizer.from_pretrained(protalbert_checkpoint,
                                                    do_lower_case=False)
    elif "esm2" in plm_type:
        if plm_type.split('-')[-1] == '8M':
            tokenizer = AutoTokenizer.from_pretrained(esm2_8m_checkpoint)
        elif plm_type.split('-')[-1] == '35M':      # actually same as 8M
            tokenizer = AutoTokenizer.from_pretrained(esm2_35m_checkpoint)
        elif plm_type.split('-')[-1] == '150M':     # actually same as 8M
            tokenizer = AutoTokenizer.from_pretrained(esm2_150m_checkpoint)
        elif plm_type.split('-')[-1] == '650M':     # actually same as 8M
            tokenizer = AutoTokenizer.from_pretrained(esm2_650m_checkpoint, cache_dir=cache_dir)
        elif plm_type.split('-')[-1] == '3B':       # actually same as 8M
            tokenizer = AutoTokenizer.from_pretrained(esm2_3b_checkpoint)
    elif "AMPLIFY" in plm_type:
        if plm_type == 'AMPLIFY-120M':
            tokenizer = AutoTokenizer.from_pretrained(amplify_120m_checkpoint, cache_dir=cache_dir)
        elif plm_type == 'AMPLIFY-120M-base':
            tokenizer = AutoTokenizer.from_pretrained(amplify_120m_base_checkpoint, cache_dir=cache_dir)
        elif plm_type == 'AMPLIFY-350M':
            tokenizer = AutoTokenizer.from_pretrained(amplify_350m_checkpoint, cache_dir=cache_dir)
    return tokenizer


'''
TAPE model
'''
class TAPE(nn.Module):
    def __init__(self, head_type='3MLP', plm_output='mean', finetune_plm=True):
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
ProtBert model
'''
class ProtBert(nn.Module):
    def __init__(self, head_type='3MLP', plm_output='mean', finetune_plm=True):
        super(ProtBert, self).__init__()
        self.proteinbert = AutoModel.from_pretrained(protbert_checkpoint, cache_dir=cache_dir)

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
ESM2 family
'''
class ESM2(nn.Module): 
    def __init__(self, head_type='3MLP', plm_output='mean', finetune_plm=True, esm_size='8M'):
        super(ESM2, self).__init__()
        if esm_size == '8M':
            self.checkpoint = esm2_8m_checkpoint
            self.hidden_size = 320
        elif esm_size == '35M':
            self.checkpoint = esm2_35m_checkpoint
            self.hidden_size = 480
        elif esm_size == '150M':
            self.checkpoint = esm2_150m_checkpoint
            self.hidden_size = 640
        elif esm_size == '650M':
            self.checkpoint = esm2_650m_checkpoint
            self.hidden_size = 1280
        elif esm_size == '3B':
            self.checkpoint = esm2_3b_checkpoint
            self.hidden_size = 2560
        else:
            raise ValueError(f"Wrong size of ESM2: {esm_size}")
        # self.esm = AutoModel.from_pretrained(self.checkpoint)
        self.esm = AutoModel.from_pretrained(self.checkpoint, cache_dir=cache_dir)
        self.head_type = head_type
        self.plm_output = plm_output
        self.finetune_plm = finetune_plm
        print(self.plm_output, self.hidden_size)

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


'''
AMPLIFY family
'''
class AMPLIFY(nn.Module): 
    def __init__(self, head_type='3MLP', plm_output='mean', finetune_plm=True, amplify_type='AMPLIFY-120M'):
        super(AMPLIFY, self).__init__()
        if amplify_type in ['AMPLIFY-120M', 'AMPLIFY-120M-base']:
            self.hidden_size = 640
        elif amplify_type in ['AMPLIFY-350M', 'AMPLIFY-350M-base']:
            self.hidden_size = 960
        else:
            raise ValueError(f"Wrong type of AMPLIFY: {amplify_type}")
        self.encoder = AutoModel.from_pretrained(
            f"chandar-lab/{amplify_type.replace('-', '_')}", trust_remote_code=True, cache_dir=cache_dir)
        self.head_type = head_type
        self.plm_output = plm_output
        self.finetune_plm = finetune_plm
        print(self.plm_output, self.hidden_size)

        # Freeze the parameters of the PLM if finetune_plm is False
        if not finetune_plm:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        # print(f"Trainable encoder parameters: {trainable_params:,}")
        
        if head_type == '3MLP':
            self.projection = MLP(self.hidden_size, [256, 64, 2])  ## 3layers
        elif head_type == '5MLP':
            self.projection = MLP(self.hidden_size, [1024, 512, 128, 32, 2])  ## 5layers

    def forward(self, input_ids):
        outputs = self.encoder(input_ids, output_hidden_states=True)
        if self.plm_output == 'mean':
            outputs = outputs.hidden_states[-1].mean(dim=1)  # [batch_size, hidden_size]
            outputs = self.projection(outputs)
        elif self.plm_output == 'cls':
            outputs = outputs.hidden_states[-1][:, 0]
            # print(outputs.shape) #!
            outputs = self.projection(outputs)

        return outputs.view(-1, outputs.size(-1))

