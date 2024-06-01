import torch.nn as nn


class MLP(nn.Module):

    def __init__(self,
                 in_dim,
                 layers,
                 activation='relu',
                 dropout=0,
                 batch_norm=True):
        super(MLP, self).__init__()
        self.mlp_layers = nn.ModuleList()
        for i, out_dim in enumerate(layers):
            self.mlp_layers.append(nn.Linear(in_dim, out_dim))
            if activation == "relu":
                self.mlp_layers.append(nn.ReLU())
            elif activation == "sigmoid":
                self.mlp_layers.append(nn.Sigmoid())
            elif activation == "tanh":
                self.mlp_layers.append(nn.Tanh())
            else:
                raise ValueError("Activation function not supported.")
            if dropout > 0:
                self.mlp_layers.append(nn.Dropout(dropout))
            if batch_norm and i != len(layers) - 1:
                self.mlp_layers.append(nn.BatchNorm1d(out_dim))
            in_dim = out_dim

    def forward(self, x):
        for layer in self.mlp_layers:
            x = layer(x)
        return x


#model = MLP(64, [32, 16], 'relu', 0.2)
#print(model)
