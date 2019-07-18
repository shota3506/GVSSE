import torch
import torch.nn as nn
import torch.nn.functional as F
from .gru import GRU
from .lstm import LSTM
from .transformer import TransformerEncoder
from .max_polling import MaxPooling


class SentenceEncoder(nn.Module):

    def __init__(self, vocab, encoder_name, d_model,
                 n_layers=None, n_head=None, d_k=None, d_v=None, d_inner=None, variance=True, eps=1e-7):
        super(SentenceEncoder, self).__init__()

        self.embed = nn.Embedding.from_pretrained(vocab.vectors)
        self.d_word = self.embed.embedding_dim
        self.encoder_name = encoder_name
        self.eps = eps

        if encoder_name == 'GRU':
            assert n_layers
            self.mean = GRU(self.d_word, d_model, n_layers)
            self.var = GRU(self.d_word, d_model, n_layers) if variance else None
        elif encoder_name == 'LSTM':
            assert n_layers
            self.mean = LSTM(self.d_word, d_model, n_layers)
            self.var = LSTM(self.d_word, d_model, n_layers) if variance else None
        elif encoder_name == 'Transformer':
            assert n_layers and n_head and d_k and d_v and d_inner
            self.mean = TransformerEncoder(self.d_word, d_model, n_layers, n_head, d_k, d_v, d_inner)
            self.var = TransformerEncoder(self.d_word, d_model, n_layers, n_head, d_k, d_v, d_inner) if variance else None
        elif encoder_name == 'MaxPooling':
            self.mean = MaxPooling(self.d_word, d_model)
            self.var = MaxPooling(self.d_word, d_model) if variance else None
        else:
            raise ValueError

        nn.Softplus()

    def forward(self, src_seq, src_pos):
        enc_seq = self.embed(src_seq)
        mean = self.mean(enc_seq, src_pos)
        if self.var:
            var = F.softplus(self.var(enc_seq, src_pos)) + self.eps
        else:
            var = torch.ones_like(mean)
        return mean, var
