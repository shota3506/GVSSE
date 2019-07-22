import torch
import torch.nn as nn
import torch.nn.functional as F


class Word2VecEncoder(nn.Module):
    def __init__(self, d_word, d_model):
        super(Word2VecEncoder, self).__init__()
        self.d_word = d_word
        self.d_model = d_model

        self.mlp = nn.Sequential(
            nn.Linear(d_word, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, enc_seq, src_pos):
        maxpooled, _ = torch.max(enc_seq, dim=1)
        return self.mlp(maxpooled)
