__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

try:
    from .layers.PatchTST_backbone import PatchTST_backbone
except:
    from layers.PatchTST_backbone import PatchTST_backbone
try:
    from .layers.PatchTSTlayers import series_decomp
except:
    from layers.PatchTSTlayers import series_decomp


class Model(nn.Module):
    def __init__(self, cin,seq_len,pred_len,nlayers,nheads,dmodel,d_ff,dropout=0.2,fc_dropout=0.2,head_dropout=0,patch_len=12,stride=8,configs = None, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        self.c_in = cin
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.nlayers = nlayers
        self.n_head = nheads
        self.d_model = dmodel
        self.d_ff = d_ff
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.head_dropout = head_dropout
        
        self.individual = 0
    
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = 'end'
        
        self.revin = 1
        self.affine = 0
        self.subtract_last = 0
        
        self.decomposition = 0
        self.kernel_size = 25
        
        
        # model
        if self.decomposition:
            self.decomp_module = series_decomp(self.kernel_size)
            self.model_trend = PatchTST_backbone(c_in=self.c_in, context_window = self.seq_len, target_window=self.pred_len, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=self.nlayers, d_model=self.d_model,
                                  n_heads=self.n_head, d_k=d_k, d_v=d_v, d_ff=self.d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=self.head_dropout, padding_patch = self.padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=self.individual, revin=self.revin, affine=self.affine,
                                  subtract_last=self.subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=self.c_in, context_window = self.seq_len, target_window=self.pred_len, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=self.nlayers, d_model=self.d_model,
                                  n_heads=self.n_head, d_k=d_k, d_v=d_v, d_ff=self.d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=self.head_dropout, padding_patch = self.padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=self.individual, revin=self.revin, affine=self.affine,
                                  subtract_last=self.subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=self.c_in, context_window = self.seq_len, target_window=self.pred_len, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=self.nlayers, d_model=self.d_model,
                                  n_heads=self.n_head, d_k=d_k, d_v=d_v, d_ff=self.d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=self.head_dropout, padding_patch = self.padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=self.individual, revin=self.revin, affine=self.affine,
                                  subtract_last=self.subtract_last, verbose=verbose, **kwargs)
            self.model.double()
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x
    