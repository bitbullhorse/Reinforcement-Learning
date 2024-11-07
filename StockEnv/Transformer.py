from torch import Tensor
from torch.nn import LayerNorm, init, TransformerDecoderLayer, TransformerDecoder, TransformerEncoderLayer, \
    TransformerEncoder
from torch.nn.modules import transformer
import string
from typing import List, Optional
import torch
import torch.nn as nn
import torch.optim as optim

import math
import torch.nn.functional as F
from layers.Embed import DataEmbedding


torch.autograd.set_detect_anomaly(True)
device = 'cuda'


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2)) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # 添加到模块的缓冲区中，不会被视为模型参数
        self.register_buffer('pe', pe)

    def forward(self, x, batch_first=True):
        if batch_first:
            # x 的形状为 (batch_size, sequence_length, d_model)
            x = x + self.pe[:, :x.size(1), :]
        else:
            # x 的形状为 (sequence_length, batch_size, d_model)
            x = x + self.pe[:, :x.size(0), :]
        return self.dropout(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        init.normal_(m.weight, mean=0, std=0.01)
    elif isinstance(m, nn.LayerNorm):
        init.constant_(m.bias, 0)
        init.constant_(m.weight, 1.0)

class TransformerCp(nn.Module):
    def __init__(self, d_model, nhead, batch_first=True, dtype=torch.float64, seq_len=12, nlayers=6, d_ff=512):
        super(TransformerCp, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.seq_len = seq_len
        self.d_ff = d_ff
        self.position_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=seq_len)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, batch_first=batch_first, dtype=dtype,
                                          num_encoder_layers=nlayers, num_decoder_layers=nlayers, dim_feedforward=d_ff)
        self.transformer.apply(init_weights)
        self.linear = nn.Linear(d_model, 512, dtype=dtype, )
        self.linear2 = nn.Linear(512, 1, dtype=dtype, )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(512).double()  # 添加 Layer Normalization

    def forward(self, src, tgt, has_mask=True):
        if has_mask:
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.shape[1], dtype=torch.float64)
            tgt_mask = tgt_mask.to(device)
        else:
            tgt_mask = None
        src = self.position_encoder(src)
        tgt = self.position_encoder(tgt)
        output = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask)[:,-1,:]
        output = self.linear(output)
        output = self.layer_norm(output)
        output = self.relu(output)
        output = self.linear2(output)
        return output

class EmbeddingTransformerCp(nn.Module):
    def __init__(self, enc_in, d_model, nhead, batch_first=True, nlayers=6, dtype=torch.float64,d_ff=2048, dropout=0.1, d_out=1, use_norm=False):
        super(EmbeddingTransformerCp, self).__init__()
        self.d_model = d_model
        self.n_head = nhead
        self.nlayers = nlayers
        self.d_ff = d_ff
        self.enc_in = enc_in

        self.use_norm = use_norm

        self.enc_embedding = DataEmbedding(enc_in, d_model)
        self.dec_embedding = DataEmbedding(enc_in, d_model)

        encoder_layer = TransformerEncoderLayer(d_model, nhead, batch_first=batch_first, dtype=dtype, dropout=dropout, dim_feedforward=d_ff)
        encoder_norm = LayerNorm(d_model,dtype=dtype)
        self.encoder = TransformerEncoder(encoder_layer, nlayers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, batch_first=batch_first, dtype=dtype, dropout=dropout, dim_feedforward=d_ff)
        decoder_norm = LayerNorm(d_model, dtype=dtype)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=nlayers, norm=decoder_norm)
        self.projector = nn.Linear(d_model, d_out, dtype=dtype)

    def forward(self, src, tgt, has_mask=True):
        if self.use_norm:
            means_src = src.mean(1, keepdim=True).detach()
            means_tgt = tgt.mean(1, keepdim=True).detach()
            src = src - means_src
            tgt = tgt - means_tgt
            std_src = torch.sqrt(torch.var(src, dim=1, keepdim=True, unbiased=False) + 1e-5)
            std_tgt = torch.sqrt(torch.var(tgt, dim=1, keepdim=True, unbiased=False) + 1e-5)
            src = src / std_src
            tgt = tgt / std_tgt

        if has_mask:
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.shape[1], dtype=torch.float64)
            tgt_mask = tgt_mask.to(device)
        else:
            tgt_mask = None
        enc_out = self.enc_embedding(src,)
        enc_out = self.encoder(enc_out,)

        dec_out = self.dec_embedding(tgt)
        dec_out = self.decoder(dec_out, enc_out, tgt_mask=tgt_mask)[:, -1, :]
        dec_out = self.projector(dec_out)

        return dec_out

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, d_model, nhead, batch_first=True, nlayers=6, dtype=torch.float64, d_ff=2048, dropout=0.1, d_out=1,):
        super(DecoderOnlyTransformer, self).__init__()
        self.d_model = d_model
        self.n_head = nhead
        self.nlayers = nlayers
        self.d_ff = d_ff

        decoder_layer = TransformerDecoderLayer(d_model, nhead, batch_first=batch_first, dtype=dtype, dropout=dropout, dim_feedforward=d_ff)
        decoder_norm = LayerNorm(d_model, dtype=dtype)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=nlayers, norm=decoder_norm)
        self.projector = nn.Linear(d_model, d_out, dtype=dtype)
    
    def forward(self, src, tgt, has_mask=False):
        if has_mask:
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.shape[1], dtype=torch.float64)
            tgt_mask = tgt_mask.to(device)
        else:
            tgt_mask = None
        dec_out = self.decoder(tgt, src, tgt_mask=tgt_mask)[:, -1, :]
        dec_out = self.projector(dec_out)
        return dec_out

class EncoderOnlyTransformer(nn.Module):
    def __init__(self, d_model, nhead, batch_first=True, nlayers=6, dtype=torch.float64, d_ff=2048, dropout=0.1, d_out=1,):
        super(EncoderOnlyTransformer, self).__init__()
        self.d_model = d_model
        self.n_head = nhead
        self.nlayers = nlayers
        self.d_ff = d_ff

        encoder_layer = TransformerEncoderLayer(d_model, nhead, batch_first=batch_first, dtype=dtype, dropout=dropout, dim_feedforward=d_ff)
        encoder_norm = LayerNorm(d_model, dtype=dtype)
        self.encoder = TransformerEncoder(encoder_layer, nlayers, encoder_norm)
        self.projector = nn.Linear(d_model, d_out, dtype=dtype)
    
    def forward(self, src, tgt, has_mask=False):
        enc_out = self.encoder(src)
        enc_out = self.projector(enc_out)
        return enc_out

class PureLSTMRegression(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True, dtype=torch.float64, num_layers=8, dropout=0):
        super(PureLSTMRegression, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first, num_layers=num_layers,dtype=dtype, dropout=dropout)
        self.Linear = nn.Linear(input_size, input_size, bias=False, dtype=dtype)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        h0, c0 = h0.to(device), c0.to(device)
        x, hidden = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.Linear(x)
        return x
