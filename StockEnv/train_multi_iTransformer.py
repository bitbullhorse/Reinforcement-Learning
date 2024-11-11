import copy
import time
from tqdm import tqdm  # 添加 tqdm 库
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
from iTransformer_model import iTransformer

import config

torch.autograd.set_detect_anomaly(True)

data_path = config.DATA_PATH
index = config.INDEX
stock_names = os.listdir(data_path)

train_end = 0.8
eval_end = 0.9

criterion_CrossEntropy = nn.CrossEntropyLoss()
criterion_NLLLoss = nn.NLLLoss()
criterion_PoissonNLLLoss = nn.PoissonNLLLoss()
criterion_L1Loss = nn.L1Loss()
criterion_MSELoss = nn.MSELoss()

price_dict = {}
def price_split(price, train_ratio=0.8, val_ratio=0.9):
    price = price[index]
    price = price.set_index('日期_Date')
    price = price.sort_index()

    n = len(price)
    train_end = int(n * train_ratio)
    val_end = int(n * val_ratio)
    train_price = price[:train_end]
    eval_price = price[train_end:val_end]
    test_price = price[val_end:]

    return train_price, eval_price, test_price

def log_training_info(model_name, end_time, test_losses, loss_func, predlen=None, model=None):
    log_path = os.path.join('/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/multi_cp', f'{model_name}_log.txt')
    end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))  # 转换时间格式
    with open(log_path, 'a') as log_file:
        log_file.write(f'Model: {model_name}, settings:dmodel:{model.d_model}, nhead:{model.n_head} nlayers:{model.nlayers} seqlen:{model.seq_len} d_ff:{model.d_ff} pred_len:{model.pred_len}\n\
                                End Time: {end_time_str}, Loss: {test_losses}, Loss_func: {(str(type(loss_func))[8:-2]).split(".")[-1]}\n')

def train_itransformer_multi_cp(epochs, model: nn.Module, CustomiTransformerDataset, price_dict, seqlen=12, predlen=5, batch_size=16, name='cp', loss_func=criterion_MSELoss):
    model.to(device)
    start_time = time.time()  # 记录开始时间
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for stock_name, price in tqdm(price_dict.items(), desc=f'Epoch {epoch+1}/{epochs} - Training'):
            train_price, eval_price, test_price = price_split(price)
            dataset_train = CustomiTransformerDataset(train_price, seqlen, predlen)
            train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
            
            for inputs, targets in train_dataloader:
                optimizer.zero_grad()
                outputs = model(inputs.to(device))
                loss = loss_func(outputs[:,:,0], targets[:,:,0].to(device))
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
        
        scheduler.step()
        
        model.eval()
        eval_losses = []
        with torch.no_grad():
            for stock_name, price in tqdm(price_dict.items(), desc=f'Epoch {epoch+1}/{epochs} - Evaluating'):
                train_price, eval_price, test_price = price_split(price)
                dataset_eval = CustomiTransformerDataset(eval_price, seqlen, predlen)
                eval_dataloader = DataLoader(dataset_eval, batch_size=batch_size, shuffle=True)
                
                for inputs, targets in eval_dataloader:
                    outputs = model(inputs.to(device))
                    loss = loss_func(outputs[:,:,0], targets[:,:,0].to(device))
                    eval_losses.append(loss.item())
        
        avg_eval_loss = np.mean(eval_losses)
        if avg_eval_loss < best_loss:
            best_loss = avg_eval_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {np.mean(train_losses)}, Eval Loss: {avg_eval_loss}')
    
    model.load_state_dict(best_model_wts)
    
    # Save the best model weights to disk
    torch.save(best_model_wts, f'/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/multi_cp/{name}_{pred_len}.pth')
    
    test_losses = {}
    model.eval()
    loss_func = nn.L1Loss()
    with torch.no_grad():
        for stock_name, price in tqdm(price_dict.items(), desc='Testing'):
            train_price, eval_price, test_price = price_split(price)
            dataset_test = CustomiTransformerDataset(test_price, seqlen, predlen)
            test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
            
            stock_test_loss = 0
            for inputs, targets in test_dataloader:
                outputs = model(inputs.to(device))
                loss = loss_func(outputs[:,:,0], targets[:,:,0].to(device))
                stock_test_loss+=loss.item()
            
            test_losses[stock_name] = stock_test_loss
    end_time = time.time()  # 记录结束时间
    print('Test Losses:', test_losses)
    print(f'Training time: {end_time - start_time} seconds')  # 打印训练时间
    log_training_info(name, end_time, test_losses, loss_func, predlen, model)

for stock_name in stock_names:
    file_path = data_path + stock_name + '/'
    file_names = os.listdir(file_path)
    price = pd.DataFrame()
    for file_name in file_names:
        xls_path = file_path + file_name
        sheet_name = pd.ExcelFile(xls_path).sheet_names[0]
        new_price = pd.read_excel(xls_path, sheet_name=sheet_name, header=0)
        price = pd.concat([price, new_price])
    price_dict[stock_name] = price


if __name__ == '__main__':
    dmodel = 256
    seqlen=12
    batch_size = 16
    EPOCHS = 200
    start_len = 5
    end_len = 11
    pred_lens = [1, 5, 10]
    itransformer_multi_cp = iTransformer(seqlen, 1, dmodel, 8, 6, 2048)
    for pred_len in pred_lens:
        model = iTransformer(d_model=dmodel, n_head=8, nlayers=6, seq_len=seqlen, d_ff=2048, pred_len=pred_len)
        train_itransformer_multi_cp(EPOCHS, model, CustomiTransformerDatasetCp, price_dict, seqlen, pred_len, batch_size, name='iTransformer_multi_cp', loss_func=criterion_MSELoss)

