import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder

from layers.Embed import DataEmbedding_inverted
from torch.nn.modules import transformer

class iTransformer(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, n_head, nlayers, d_ff,use_norm=True):
        super(iTransformer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.nlayers = nlayers
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_ff = d_ff
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model,)
        encoder_layers = TransformerEncoderLayer(d_model, n_head, d_ff, dropout=0.1, batch_first=True, dtype=torch.float64)
        self.encoder = TransformerEncoder(encoder_layers, num_layers=self.nlayers,)
        self.projector = nn.Linear(self.d_model, self.pred_len, bias=True, dtype=torch.float64)
        self.use_norm=use_norm

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out = self.encoder(enc_out)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
