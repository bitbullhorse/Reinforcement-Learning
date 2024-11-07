import numpy as np
import scipy.stats as st
from numpy import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import os
from arch import arch_model
from Transformer import *
from datautil import *
from torch.utils.data import DataLoader as Dataloader
from train_func import train_iTranformer, train_transformer

from iTransformer_model import iTransformer

torch.autograd.set_detect_anomaly(True)

sheet_name = 'DRESSTK_2021_'

stock_num = '000001/'

file_path = '/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/股票数据/' + stock_num
file_names = os.listdir(file_path)
plt.figure(figsize=(15, 10))
df = pd.DataFrame()
price = pd.DataFrame()
for file_name in file_names:
    xls_path = file_path + file_name
    sheet_name = pd.ExcelFile(xls_path).sheet_names[0]
    # 确认列索引是否在范围内
    new_price = pd.read_excel(xls_path, sheet_name=sheet_name, header=0)
    price = pd.concat([price, new_price])


index = ['日期_Date', '收盘价_Clpr', '开盘价_Oppr', '最高价_Hipr', '最低价_Lopr', '复权价1(元)_AdjClpr1', '复权价2(元)_AdjClpr2','成交量_Trdvol',\
               '成交金额_Trdsum','日振幅(%)_Dampltd','总股数日换手率(%)_DFulTurnR', '流通股日换手率(%)_DTrdTurnR', '日收益率_Dret', '日资本收益率_Daret',\
               '等权平均市场日收益率_Dreteq','流通市值加权平均市场日收益率_Drettmv', '总市值加权平均市场日收益率_Dretmc', '等权平均市场日资本收益率_Dareteq',\
               '总市值加权平均日资本收益_Daretmc', '日无风险收益率_DRfRet', '市盈率_PE']

criterion_CrossEntropy = nn.CrossEntropyLoss()
criterion_NLLLoss = nn.NLLLoss()
criterion_PoissonNLLLoss = nn.PoissonNLLLoss()
criterion_L1Loss = nn.L1Loss()
criterion_MSELoss = nn.MSELoss()

price = price[index]
price = price.set_index('日期_Date')
price = price.sort_index()

start_date = str(price.index[0])[0:-9]
end_date = str(price.index[-1])[0:-9]
print('start_date:', start_date)
print('end_date:', end_date)

n = len(price)
train_end = int(n * 0.8)
val_end = int(n * 0.9)
train_price = price[:train_end]
eval_price = price[train_end:val_end]
test_price = price[val_end:]

def train_itransformer_cp(epochs, model, CustomiTransformerDataset, seqlen=12, predlen=5, batch_size=16, name='cp', loss_func=criterion_MSELoss):
    model.to(device)

    dataset_train = CustomiTransformerDataset(train_price, seqlen, predlen)
    dataset_eval = CustomiTransformerDataset(eval_price, seqlen, predlen)
    dataset_test = CustomiTransformerDataset(test_price, seqlen, predlen)

    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(dataset_eval, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(dataset_test, batch_size=1, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)

    train_iTranformer(model, epochs, train_dataloader, loss_func, optimizer, scheduler, eval_dataloader,
                      test_dataloader, stock_num[0:-1], name)

def train_transformer_cp(model, epochs, CustomDataset, seqlen=12, batch_size=16, name='cp', predlen=1, loss_func=criterion_MSELoss):
    model.to(device)

    train_data = CustomDataset(train_price, seqlen, predlen)
    eval_data = CustomDataset(eval_price, seqlen, predlen)
    test_data = CustomDataset(test_price, seqlen, predlen)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    optimizer_tf = optim.Adam(model.parameters(), lr=0.001)
    scheduler_tf = optim.lr_scheduler.StepLR(optimizer_tf, step_size=30, gamma=0.2)

    train_transformer(model, epochs, train_loader, loss_func, optimizer_tf, scheduler_tf, eval_loader,
                   test_loader, stock_num[0:-1], name, predlen)

if __name__ == '__main__':
    dmodel = 128
    seqlen=12
    batch_size = 16
    EPOCHS = 200

    for predlen in range(1, 11):
        iTransformer_cp = iTransformer(seqlen, predlen, dmodel, 8, 6, 512)
        train_itransformer_cp(EPOCHS, iTransformer_cp, CustomiTransformerDatasetCp, seqlen, predlen, loss_func = criterion_L1Loss)
    for predlen in range(2, 11):
        # embdtrans = EmbeddingTransformerCp(len(index) - 1, dmodel, 8, True, 6, d_ff=2048)
        # train_transformer_cp(embdtrans, EPOCHS, CustomDatasetCp, seqlen, batch_size, 'cp', predlen, criterion_MSELoss)
        # transformer_cp = TransformerCp(len(index) - 1, 4, True, torch.float64, seqlen, 6, 512)
        # train_transformer_cp(transformer_cp, EPOCHS, CustomDatasetCp, seqlen, batch_size, 'cp', predlen, criterion_L1Loss)
        # DecoderOnlyTransformer_cp = DecoderOnlyTransformer(len(index) - 1, 4, True, 6, d_ff=512)
        # train_transformer_cp(DecoderOnlyTransformer_cp, EPOCHS, CustomDatasetCp, seqlen, batch_size, 'cp', predlen, criterion_L1Loss)
        EncoderOnlyTransformer_cp = EncoderOnlyTransformer(len(index) - 1, 4, True, 6, d_ff=512)
        train_transformer_cp(EncoderOnlyTransformer_cp, EPOCHS, CustomDatasetCp, seqlen, batch_size, 'cp', predlen, criterion_L1Loss)
