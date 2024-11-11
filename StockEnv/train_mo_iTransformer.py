import copy
import os
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import config
from datautil import *
from iTransformer_model import multi_iTransformer
from iTransformer_model import multi_iTransformer_Dec
from iTransformer_model import multi_iTransformer_multi_Dec
import time
import random  # 添加随机模块


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


def train_mo_itransformer(epochs, model: nn.Module, CustomiTransformerDataset, price_dict, seqlen=12, predlen=5, batch_size=16, name='cp', loss_func=criterion_MSELoss, l2_lambda=0.01):
    model.to(device)
    start_time = time.time()  # 记录开始时间
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_lambda)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
    best_loss = float('inf')
    try:
        for epoch in range(epochs):
            model.train()
            train_losses = []
            # 在每个 epoch 开始时，获取并打乱股票数据列表
            stock_list = list(price_dict.items())
            random.shuffle(stock_list)
            for stock_name, price in tqdm(stock_list, desc=f'Epoch {epoch+1}/{epochs} - Training'):
                train_price, _, _ = price_split(price)
                train_data = CustomiTransformerDataset(train_price, seqlen, pred_len)
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                train_loss = 0
                for inputs, target in train_loader:
                    optimizer.zero_grad()
                    output = model(inputs, stock_name)
                    loss = loss_func(output[:, :, 0], target[:, :, 0])
                    train_loss += loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                train_losses.append(train_loss)
            scheduler.step()
            model.eval()
            eval_losses = []
            with torch.no_grad():
                for stock_name, price in tqdm(price_dict.items(), desc=f'Epoch {epoch+1}/{epochs} - Evaluating'):
                    _, eval_price, _ = price_split(price)
                    eval_data = CustomiTransformerDataset(eval_price, seqlen, predlen)
                    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=True)
                    loss_sum = 0
                    for inputs, target in eval_loader:
                        output = model(inputs, stock_name)
                        loss = loss_func(output[:, :, 0], target[:, :, 0])
                        loss_sum += loss.item()
                    eval_losses.append(loss_sum)                
            avg_eval_loss = np.mean(eval_losses)
            if avg_eval_loss < best_loss:
                best_loss = avg_eval_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {np.mean(train_losses)}, Eval Loss: {avg_eval_loss}')
        model.load_state_dict(best_model_wts)
        torch.save(best_model_wts, f'/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/multi_cp/{name}_{pred_len}.pth')
    except KeyboardInterrupt:
        print('Training interrupted by user.')
    test_losses = {}
    model.eval()
    loss_func = criterion_L1Loss
    with torch.no_grad():
        for stock_name, price in tqdm(price_dict.items(), desc='Testing'):
            _, _, test_price = price_split(price)
            test_data = CustomiTransformerDataset(test_price, seqlen, predlen)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
            stock_test_loss = 0
            for inputs, target in test_loader:
                output = model(inputs, stock_name)
                loss = loss_func(output[:, :, 0], target[:, :, 0])
                stock_test_loss += loss.item()
            test_losses[stock_name] = stock_test_loss

    end_time = time.time()
    print(f'Training time: {end_time - start_time}seconds')
    print(f'Test Loss: {test_losses}')
    log_training_info(name, end_time, test_losses, loss_func, predlen, model=model)

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
    pred_lens = [1]
    # for pred_len in pred_lens:
    #     model = multi_iTransformer(d_model=dmodel, n_head=8, nlayers=6, seq_len=seqlen, d_ff=4096, keys=stock_names,pred_len=pred_len)
    #     train_mo_itransformer(EPOCHS, model, CustomiTransformerDatasetCp, price_dict, seqlen=seqlen, predlen=pred_len, batch_size=batch_size, name='iTransformer_multi_out', loss_func=criterion_MSELoss)
    
    for pred_len in pred_lens:
        model = multi_iTransformer_multi_Dec(len(index) - 1,d_model=dmodel, n_head=8, nlayers=6, seq_len=seqlen, d_ff=4096, keys=stock_names,pred_len=pred_len)
        train_mo_itransformer(EPOCHS, model, CustomiTransformerDatasetCp, price_dict, seqlen=seqlen, predlen=pred_len, batch_size=batch_size, name='iTransformer_multi_dec', loss_func=criterion_MSELoss)
