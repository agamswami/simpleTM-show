import torch
import torch.nn as nn

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1): #c_in: Usually, this represents the number of channels/variables. In SimpleTM, however, it gets dynamically passed the seq_len (e.g., 96 steps) during model creation. d_model: The target dimension for the embedding.
        super(DataEmbedding_inverted, self).__init__()                            #
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        return self.dropout(x)