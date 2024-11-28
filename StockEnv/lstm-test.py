import os
import pandas as pd
import torch
import torch.nn as nn
from datautil import CustomiTransformerDatasetCp
from funcutil import train_itransformer_cp

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.seq_len = 12
        self.pred_len = 1

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.unsqueeze(1)
    
file_path = '/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/股票数据/000001'
file_names = os.listdir(file_path)
price = pd.DataFrame()
for file_name in file_names:
    xls_path = file_path + '/' + file_name
    sheet_name = pd.ExcelFile(xls_path).sheet_names[0]
    # 确认列索引是否在范围内
    new_price = pd.read_excel(xls_path, sheet_name=sheet_name, header=0)
    price = pd.concat([price, new_price])

loss_func = nn.L1Loss()
model = LSTMPredictor(input_size=20, hidden_size=128, num_layers=6, output_size=20)
model.double()
train_itransformer_cp(200, model, CustomiTransformerDatasetCp, '000001/', price, 12, 1, 16, name='cp', loss_func=loss_func, model_name='LSTMPredictor')
