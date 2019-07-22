import torch
import torch.nn as nn
import torch.nn.functional as F
from .gru import GRUEncoder
from .lstm import LSTMEncoder
from .word2vec import Word2VecEncoder


class SentenceEncoder(nn.Module):

    def __init__(self, vocab, encoder_name, d_model, n_layers=None, variance=True, eps=1e-7):
        super(SentenceEncoder, self).__init__()

        self.embed = nn.Embedding.from_pretrained(vocab.vectors)
        self.d_word = self.embed.embedding_dim
        self.encoder_name = encoder_name
        self.eps = eps

        if encoder_name == 'GRU':
            assert n_layers
            self.mean = GRUEncoder(self.d_word, d_model, n_layers)
            self.var = GRUEncoder(self.d_word, d_model, n_layers) if variance else None
        elif encoder_name == 'LSTM':
            assert n_layers
            self.mean = LSTMEncoder(self.d_word, d_model, n_layers)
            self.var = LSTMEncoder(self.d_word, d_model, n_layers) if variance else None
        elif encoder_name == 'Word2Vec':
            self.mean = Word2VecEncoder(self.d_word, d_model)
            self.var = Word2VecEncoder(self.d_word, d_model) if variance else None
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
