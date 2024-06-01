import torch.nn as nn
import torch.nn.functional as F
from projection_head import MLP


class baseline_mlp(nn.Module):
    def __init__(self,
                 emb_dim,
                 seq_max_len,
                 layers=[256, 64, 2],   # same as PLM head
                 voacb_size=30,         # ProtBERT tokenizer
                 padding_idx=25,        # ProtBERT tokenizer
                 activation='relu',
                 dropout=0,
                 batch_norm=True):
        super(baseline_mlp, self).__init__()
        self.embedding = nn.Embedding(voacb_size, emb_dim, padding_idx=padding_idx)
        self.flatten = nn.Flatten()
        self.classifier = MLP(emb_dim*seq_max_len, layers,
                              activation, dropout, batch_norm)
    def forward(self, x):
        x = self.embedding(x)   # [B, seq_max_len, emb_dim]
        x = self.flatten(x)     # [B, seq_max_len*emb_dim]
        x = self.classifier(x)  # [B, out_dim]
        return x


"""
one-hot encoding
"""
# class baseline_mlp(nn.Module):
#     def __init__(self,
#                  layers,
#                  seq_max_len,
#                  voacb_size=30,     # ProtBERT tokenizer
#                  activation='relu',
#                  dropout=0,
#                  batch_norm=True):
#         super(baseline_mlp, self).__init__()
#         self.emb_dim = voacb_size
#         self.flatten = nn.Flatten()
#         self.classifier = MLP(self.emb_dim*seq_max_len, layers,
#                               activation, dropout, batch_norm)
#     def forward(self, x):
#         x = F.one_hot(x, num_classes=self.emb_dim).float()   # [B, seq_max_len, emb_dim]
#         x = self.flatten(x)     # [B, seq_max_len*emb_dim]
#         x = self.classifier(x)  # [B, out_dim]
#         return x

# model = baseline_mlp(layers=[512, 512, 128, 2], 
#                     #  layers=[256, 64, 2], 
#                      seq_max_len=input_seq_max_len)