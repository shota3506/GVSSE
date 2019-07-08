import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout=0.1):
        super(LSTMEncoder, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers,
                            dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, embedded, input_lengths, hidden=None):
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, (hidden, _) = self.lstm(packed, hidden)
        hidden = torch.mean(hidden, dim=0)
        return F.normalize(hidden)
