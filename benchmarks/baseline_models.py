import torch
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


class baseline_rnn(nn.Module):
    def __init__(self, emb_dim, hidden_size, output_size, rnn_type="rnn",
                 voacb_size=30, padding_idx=25, 
                 use_bidirection=False, use_LN=False, pooling_type="average",
                 dropout_p=0, num_layers=1):
        super(baseline_rnn, self).__init__()

        # Embedding: token to vector
        self.embedding = nn.Embedding(voacb_size, emb_dim, padding_idx=padding_idx)

        # RNN layer
        self.direction = 2 if use_bidirection else 1
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, 
                            dropout=dropout_p, bidirectional=use_bidirection)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(input_size=emb_dim, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, 
                            dropout=dropout_p, bidirectional=use_bidirection)
        else:
            self.rnn = nn.RNN(input_size=emb_dim, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, 
                            dropout=dropout_p, bidirectional=use_bidirection)
        ## output: [B, Length, D∗H] when batch_first=True
        ## h_n(final hidden state): [D∗num_layers, B, H]

        # LayerNorm layer
        self.use_LN = use_LN
        if self.use_LN:
            self.layer_norm = nn.LayerNorm(hidden_size*self.direction)

        # Fully connected layer
        self.pooling = pooling_type     # "average", "last"
        self.fc = nn.Linear(hidden_size*self.direction, output_size)

    def forward(self, x):

        x = self.embedding(x)   # [B, seq_max_len, emb_dim]
        out, _ = self.rnn(x)    # [B, seq_max_len, hidden_size]

        if self.use_LN:
            out = self.layer_norm(out)

        if self.pooling == "average":
            # Global average pooling
            out = torch.mean(out, dim=1)    # [B, hidden_size]
        elif self.pooling == "last":
            # Last output
            out = out[:, -1]                # [B, hidden_size]

        out = self.fc(out)                  # [B, output_size]
        return out


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