import torch
from torch import nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class Embedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(Embedding, self).__init__()
        self.token_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.positional_embedding = PositionalEmbedding(d_model=d_model)

    def forward(self, x):
        return self.token_embedding(x) + self.positional_embedding(x)


class Model(nn.Module):
    def __init__(self, input_size, d_model=512, n_heads=8, num_enc_layers=6, num_dec_layers=6, batch_first=True):
        super(Model, self).__init__()
        self.n_heads = n_heads
        self.transformer = nn.Transformer(d_model=d_model, batch_first=batch_first, nhead=n_heads,
                                          num_encoder_layers=num_enc_layers, num_decoder_layers=num_dec_layers)
        self.enc_embedding = Embedding(c_in=input_size, d_model=d_model)
        self.dec_embedding = Embedding(c_in=input_size, d_model=d_model)
        self.linear = nn.Linear(d_model, input_size)

    def forward(self, src, placeholder1, tgt, placeholder2):
        # src = torch.tile(src, (self.n_heads, 1))
        # tgt = torch.tile(tgt, (self.n_heads, 1))
        src = self.enc_embedding(src)
        tgt = self.dec_embedding(tgt)
        output = self.transformer(src, tgt)
        output = self.linear(output)
        return output
