from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pandas.core.frame
import torch
import numpy as np

INDEX = ['收盘价_Clpr', '开盘价_Oppr', '最高价_Hipr', '最低价_Lopr', '复权价1(元)_AdjClpr1', '复权价2(元)_AdjClpr2', '成交量_Trdvol',\
               '成交金额_Trdsum','日振幅(%)_Dampltd','总股数日换手率(%)_DFulTurnR', '流通股日换手率(%)_DTrdTurnR', '日收益率_Dret', '日资本收益率_Daret', \
               '等权平均市场日收益率_Dreteq','流通市值加权平均市场日收益率_Drettmv', '总市值加权平均市场日收益率_Dretmc', '等权平均市场日资本收益率_Dareteq',\
               '总市值加权平均日资本收益_Daretmc', '日无风险收益率_DRfRet', '市盈率_PE']
device = 'cuda'


# Transformer模型 - 预测收盘价
class CustomDatasetCp(Dataset):
    def __init__(self, dataframe: pandas.core.frame.DataFrame, seq_size: int = 30, pred_len:int = 1):
        self.seq_size = seq_size
        self.pred_len = pred_len
        tmp_df = dataframe.copy()
        tmp_df = tmp_df.dropna()
        minmaxscaler = MinMaxScaler()
        # tmp_df[INDEX] = minmaxscaler.fit_transform(tmp_df[INDEX])
        self.seq_df = tmp_df
        self.close_df = tmp_df['收盘价_Clpr']

    def __getitem__(self, index):
        src_df = self.seq_df.iloc[index: index + self.seq_size]
        tgt_df = self.seq_df.iloc[index + self.seq_size + self.pred_len - 1]
        Cl_pr = self.close_df.iloc[index + self.seq_size + self.pred_len - 1]
        return torch.tensor(src_df.values, dtype=torch.float64, device=device), torch.unsqueeze(torch.tensor(tgt_df.values,
                                                                                             dtype=torch.float64,
                                                                                             device=device), 0), torch.unsqueeze(torch.tensor(Cl_pr, dtype=torch.float64, device=device), dim=0)
    def __len__(self):
        return int(min(len(self.seq_df) - self.seq_size - self.pred_len, len(self.close_df) - self.seq_size - self.pred_len)) + 1


# Transformer模型 - 预测收益率
class CustomDatasetRt(Dataset):
    def __init__(self, dataframe: pandas.core.frame.DataFrame, seq_size: int = 30, pred_len:int = 1):
        self.seq_size = seq_size
        self.pred_len = pred_len
        tmp_df = dataframe.copy()
        tmp_df = tmp_df.dropna()
        minmaxscaler = MinMaxScaler()
        # tmp_df[INDEX] = minmaxscaler.fit_transform(tmp_df[INDEX])
        self.seq_df = tmp_df
        self.close_df = tmp_df['收盘价_Clpr']
        self.return_df = self.close_df.pct_change()
        self.return_df[0] = 0
        self.return_df = self.return_df.dropna()
        # self.seq_df = self.seq_df[self.return_df.index]

    def __getitem__(self, index):
        src_df = self.seq_df.iloc[index: index + self.seq_size]
        tgt_df = self.seq_df.iloc[index + self.seq_size + self.pred_len - 1]
        rt = self.close_df.iloc[index + self.seq_size + self.pred_len - 1]
        return torch.tensor(src_df.values, dtype=torch.float64, device=device), torch.tensor(tgt_df.values,
                                                                                             dtype=torch.float64,
                                                                                             device=device), torch.unsqueeze(torch.tensor(rt, dtype=torch.float64, device=device), dim=0)
    def __len__(self):
        return int(min(len(self.seq_df  - self.seq_size - self.pred_len), len(self.close_df) - self.seq_size - self.pred_len)) + 1


# iTransformer模型 - 预测收盘价
class CustomiTransformerDatasetCp(Dataset):
    def __init__(self, dataframe: pandas.core.frame.DataFrame, seq_len: int=30, pred_len: int=2):
        self.seq_len = seq_len
        self.pred_len = pred_len
        tmp_df = dataframe.copy()
        minmaxscaler = MinMaxScaler()
        # tmp_df[INDEX] = minmaxscaler.fit_transform(tmp_df[INDEX])
        tmp_df = tmp_df.dropna()
        self.seq_df = tmp_df

    def __getitem__(self, index):
        seq = self.seq_df[index: index + self.seq_len]
        label = self.seq_df[index + self.seq_len: index + self.seq_len + self.pred_len]
        return torch.tensor(seq.values, dtype=torch.float64, device=device), torch.tensor(label.values, dtype=torch.float64, device=device),

    def __len__(self):
        return int(len(self.seq_df) - self.seq_len - self.pred_len + 1)


# iTransformer模型 - 预测收益率
class CustomiTransformerDatasetRt(Dataset):
    def __init__(self, dataframe: pandas.core.frame.DataFrame, seq_len: int=30, pred_len: int=2):
        self.seq_len = seq_len
        self.pred_len = pred_len
        tmp_df = dataframe.copy()
        minmaxscaler = MinMaxScaler()
        # tmp_df[INDEX] = minmaxscaler.fit_transform(tmp_df[INDEX])
        tmp_df = tmp_df.dropna()
        self.seq_df = tmp_df
        self.close_df = tmp_df['收盘价_Clpr']
        self.return_df = self.close_df.pct_change()
        self.return_df[0] = 0
        self.return_df = self.return_df.dropna()

    def __getitem__(self, index):
        seq = self.seq_df[index: index + self.seq_len]
        label = self.seq_df[index + self.seq_len: index + self.seq_len + self.pred_len]
        return torch.tensor(seq.values, dtype=torch.float64, device=device), torch.tensor(label.values, dtype=torch.float64, device=device),

    def __len__(self):
        return int(len(self.seq_df) - self.seq_len - self.pred_len + 1)

