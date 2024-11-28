import copy
import os
from matplotlib import pyplot as plt
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
from DLinear import iTransformer_multi_Dlinear, Model as DLinear
from iTransformer_model import iTransformer
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
                train_data = CustomiTransformerDataset(train_price, seqlen, predlen)
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
    except KeyboardInterrupt:
        print('Training interrupted by user.')
    torch.save(best_model_wts, f'/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/multi_cp/{name}_{predlen}.pth')
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
                loss = loss_func(output[:, -1, 0], target[:, -1, 0])
                stock_test_loss += loss.item()
            test_losses[stock_name] = stock_test_loss

    end_time = time.time()
    print(f'Training time: {end_time - start_time}seconds')
    print(f'Test Loss: {test_losses}')
    log_training_info(name, end_time, test_losses, loss_func, predlen, model=model)


def plot_result(model_name,stock_name, label, pred):
    fig, axes = plt.subplots(1, 1)
    axes.grid(True, axis='y')  # 仅添加水平网格
    axes.grid(True, axis='x')  # 仅添加垂直网格
    axes.plot(label, 'b-', label='label')
    axes.plot(pred, 'r-', label='predict')
    axes.set_title('Close Price')
    axes.set_xlabel('day')
    axes.set_ylabel('price')
    plt.legend()
    plt.subplots_adjust(hspace=0.5)
    path = '/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/multi_cp/' + model_name
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + '/' + f'{stock_name}.png')


def iTransformer_test(model: nn.Module, CustomiTransformerDataset, price_dict, seqlen=12, predlen=1, batch_size=16, name='cp', loss_func=criterion_MSELoss,):
    model.to(device)
    test_losses = {}
    model.eval()
    better_count = 0
    with torch.no_grad():
        for stock_name, price in tqdm(price_dict.items(), desc='Testing'):
            _, _, test_price = price_split(price)
            label = []
            pred = []
            test_data = CustomiTransformerDataset(test_price, seqlen, predlen)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
            stock_test_loss = 0
            length = len(test_data)
            for inputs, target in test_loader:
                output = model(inputs, stock_name)
                loss = loss_func(output[:, :, 0], target[:, :, 0])
                stock_test_loss += loss.item()
                output = output[:, -1, 0].reshape(-1).tolist()  # 使用 reshape
                target = target[:, -1, 0].reshape(-1).tolist()
                label += target
                pred += output
            test_losses[stock_name] = stock_test_loss
            if predlen == 1:
                plot_result(name, stock_name, label, pred)
            itransformer = iTransformer(d_model=128, n_head=8, nlayers=6, seq_len=seqlen, d_ff=512, pred_len=predlen)
            itransformer.load_state_dict(torch.load(f'/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/saved_model/{stock_name}/cp/iTransformer_{pred_len}.pth'))
            itransformer.to(device)
            itransformer.eval()
            test_loss = 0
            for inputs, target in test_loader:
                output = itransformer(inputs)
                loss = loss_func(output[:, :, 0], target[:, :, 0])
                test_loss += loss.item()
            # dlinear = DLinear(20, 36, 1, 0)
            # dlinear.load_state_dict(torch.load(f'/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/saved_model/{stock_name}/cp/DLinear_1.pth'))
            # dlinear.to(device)
            # dlinear.eval()
            # dlinear.double()
            # test_loss = 0
            # test_data = CustomiTransformerDataset(test_price, 36, 1)
            # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
            # for inputs, target in test_loader:
            #     output = dlinear(inputs)
            #     loss = loss_func(output[:, :, 0], target[:, :, 0])
            #     test_loss += loss.item()
            
            if stock_test_loss / length < test_loss / len(test_data):
                better_count += 1

    print(f'Test Loss: {test_losses}')
    log_training_info(name, time.time(), test_losses, loss_func, predlen, model=model)
    return test_losses, better_count


def train_mo_itransformer_random_data(epochs, model: nn.Module, CustomiTransformerDataset, price_dict, seqlen=12, predlen=1, batch_size=16, name='cp', loss_func=criterion_MSELoss, l2_lambda=0.01):
    model.to(device)
    start_time = time.time()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_lambda)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
    best_loss = float('inf')
    n = 50  # 每次从每个数据集取出的 batch 数
    try:
        for epoch in range(epochs):
            model.train()
            # 打乱股票数据列表
            stock_list = list(price_dict.items())
            random.shuffle(stock_list)
            # 为每个股票数据集创建迭代器
            dataloader_iterators = {}
            for stock_name, price in stock_list:
                train_price, _, _ = price_split(price)
                train_data = CustomiTransformerDataset(train_price, seqlen, predlen)
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                dataloader_iterators[stock_name] = iter(train_loader)
            # 记录尚未完成的数据集
            remaining_stocks = set(dataloader_iterators.keys())
            train_losses = {stock_name:0 for stock_name in remaining_stocks}
            while remaining_stocks:
                for stock_name in tqdm(list(remaining_stocks), desc='Processing Stocks', leave=False):
                    data_iterator = dataloader_iterators[stock_name]
                    batch_count = 0
                    with tqdm(total=n, desc=f'Training {stock_name}', leave=False) as pbar:
                        while batch_count < n:
                            try:
                                inputs, target = next(data_iterator)
                                optimizer.zero_grad()
                                output = model(inputs, stock_name)
                                loss = loss_func(output[:, :, 0], target[:, :, 0])
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                optimizer.step()
                                batch_count += 1
                                train_losses[stock_name] += loss.item()
                                pbar.update(1)
                            except StopIteration:
                                # 数据集遍历完毕，移除
                                remaining_stocks.remove(stock_name)
                                break
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
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {np.mean(list(train_losses.values()))}, Eval Loss: {avg_eval_loss}')
    except KeyboardInterrupt:
        print('Training interrupted by user.')
    torch.save(best_model_wts, f'/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/multi_cp/{name}_{predlen}.pth')
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


def train_multi_itransformer(epochs, model:nn.Module, CustomMultiCpDataset, train_dict, eval_dict, test_dict, seqlen=12, predlen=1, batch_size=16, name='cp', loss_func=criterion_MSELoss, l2_lambda=0.01):
    model.to(device)
    start_time = time.time()  # 记录开始时间
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_lambda)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
    best_loss = float('inf')
    checkpoint_path = f'/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/multi_cp/{name}_checkpoint.pth'
    
    # 尝试加载检查点
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch'] + 1
        print(f'Loaded checkpoint from epoch {start_epoch}')
    else:
        start_epoch = 0

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            train_data = CustomMultiCpDataset(train_dict, seqlen, predlen)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            train_loss = 0
            for inputs, target, keys in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training'):
                loss_sum = 0
                optimizer.zero_grad()
                for i, key in enumerate(keys):
                    output = model(inputs[i].unsqueeze(0), key)
                    loss = loss_func(output[:,:,0], target[i,:,0].unsqueeze(0))
                    loss_sum += loss
                loss_sum.backward()    
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss_sum.item()
            scheduler.step()
            model.eval()
            eval_losses = {}
            with torch.no_grad():
                eval_data = CustomMultiCpDataset(eval_dict, seqlen, predlen)
                eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=True)
                for inputs, target, keys in tqdm(eval_loader, desc=f'Epoch {epoch+1}/{epochs} - Evaluating'):
                    for i, key in enumerate(keys):
                        output = model(inputs[i].unsqueeze(0), key)
                        loss = loss_func(output[:,:,0], target[i,:,0])
                        if key not in eval_losses:
                            eval_losses[key] = loss.item()
                        else:
                            eval_losses[key] += loss.item()
            avg_eval_loss = np.mean(list(eval_losses.values()))
            if avg_eval_loss < best_loss:
                best_loss = avg_eval_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss
                }, checkpoint_path)  # 保存检查点
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss / len(eval_losses)}, Eval Loss: {avg_eval_loss}')
        model.load_state_dict(best_model_wts)
        torch.save(best_model_wts, f'/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/multi_cp/{name}_{predlen}.pth')
    except KeyboardInterrupt:
        print('Training interrupted by user.')
    test_losses = {}
    model.eval()
    loss_func = criterion_L1Loss
    with torch.no_grad():
        test_data = CustomMultiCpDataset(test_dict, seqlen, predlen)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        for inputs, target, keys in tqdm(test_loader, desc='Testing'):
            for i, key in enumerate(keys):
                output = model(inputs[i].unsqueeze(0), key)
                loss = loss_func(output[:,:,0], target[i,:,0].unsqueeze(0))
                if key not in test_losses:
                    test_losses[key] = loss.item()
                else:
                    test_losses[key] += loss.item()
    end_time = time.time()
    print(f'Training time: {end_time - start_time}seconds')
    print(f'Test Loss: {test_losses}')
    log_training_info(name, end_time, test_losses, loss_func, predlen, model=model)


train_dict = {}
eval_dict = {}
test_dict = {}

for stock_name in stock_names:
    file_path = data_path + stock_name + '/'
    file_names = os.listdir(file_path)
    price = pd.DataFrame()
    for file_name in file_names:
        xls_path = file_path + file_name
        sheet_name = pd.ExcelFile(xls_path).sheet_names[0]
        new_price = pd.read_excel(xls_path, sheet_name=sheet_name, header=0)
        price = pd.concat([price, new_price])
    print(f'Loaded {stock_name} data.')
    price_dict[stock_name] = price
    train_price, eval_price, test_price = price_split(price)
    train_dict[stock_name] = train_price
    eval_dict[stock_name] = eval_price
    test_dict[stock_name] = test_price


def train_multi_DLinear():
    dmodel = 256
    seqlen=36
    batch_size = 16
    EPOCHS = 200
    pred_lens = [1, 5, 10]
    for pred_len in pred_lens:
        model = iTransformer_multi_Dlinear(d_model=dmodel, n_head=8, nlayers=6, seq_len=seqlen, d_ff=4096, pred_len=pred_len, individual=0, keys=stock_names)
        train_mo_itransformer(EPOCHS, model, CustomiTransformerDatasetCp, price_dict, seqlen=seqlen, predlen=pred_len, batch_size=batch_size, name='iTransformer_multi_Dlinear', loss_func=criterion_MSELoss)
    exit(0)

# train_multi_DLinear()

if __name__ == '__main__':
    dmodel = 256
    seqlen=36
    batch_size = 16
    EPOCHS = 200
    start_len = 5
    end_len = 11
    pred_lens = [1, 5, 10]

    for pred_len in pred_lens:
        model = multi_iTransformer(d_model=dmodel, n_head=8, nlayers=6, seq_len=seqlen, d_ff=4096, keys=stock_names,pred_len=pred_len)
        train_mo_itransformer(EPOCHS, model, CustomiTransformerDatasetCp, price_dict, seqlen=seqlen, predlen=pred_len, batch_size=batch_size, name='iTransformer_multi_36_out', loss_func=criterion_MSELoss)
    # for pred_len in pred_lens:
    #     model = multi_iTransformer_multi_Dec(len(index) - 1,d_model=dmodel, n_head=8, nlayers=6, seq_len=seqlen, d_ff=4096, keys=stock_names,pred_len=pred_len)
    #     train_mo_itransformer(EPOCHS, model, CustomiTransformerDatasetCp, price_dict, seqlen=seqlen, predlen=pred_len, batch_size=batch_size, name='iTransformer_multi_dec', loss_func=criterion_MSELoss)
    exit(0)
    # for pred_len in pred_lens:
    #     model = multi_iTransformer_multi_Dec(len(index) - 1,d_model=dmodel, n_head=8, nlayers=6, seq_len=seqlen, d_ff=4096, keys=stock_names,pred_len=pred_len)
    #     train_multi_itransformer(EPOCHS, model, CustomMultiCpDataset, train_dict, eval_dict, test_dict, seqlen=seqlen, predlen=pred_len, batch_size=batch_size, name='iTransformer_multi_dec_1', loss_func=criterion_MSELoss)

    # for pred_len in pred_lens:
    #     model = multi_iTransformer_multi_Dec(len(index) - 1,d_model=dmodel, n_head=8, nlayers=6, seq_len=seqlen, d_ff=4096, keys=stock_names,pred_len=pred_len)
    #     train_mo_itransformer_random_data(EPOCHS, model, CustomiTransformerDatasetCp, price_dict, seqlen=seqlen, predlen=pred_len, batch_size=batch_size, name='iTransformer_multi_dec_random', loss_func=criterion_MSELoss)
    for pred_len in pred_lens:
        model_out = multi_iTransformer(d_model=dmodel, n_head=8, nlayers=6, seq_len=seqlen, d_ff=4096, keys=stock_names,pred_len=pred_len)
        model_out.load_state_dict(torch.load(f'/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/multi_cp/iTransformer_multi_out_{pred_len}.pth'))
        model_dec = multi_iTransformer_multi_Dec(len(index) - 1,d_model=dmodel, n_head=8, nlayers=6, seq_len=seqlen, d_ff=4096, keys=stock_names,pred_len=pred_len)
        model_dec.load_state_dict(torch.load(f'/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/multi_cp/iTransformer_multi_dec_{pred_len}.pth'))
        loss_list = [criterion_L1Loss, criterion_MSELoss]
        out_result_list = []
        dec_result_list = []
        out_result, out_count = iTransformer_test(model_out, CustomiTransformerDatasetCp, price_dict, seqlen=seqlen, predlen=pred_len, batch_size=batch_size, name='iTransformer_multi_out', loss_func=criterion_MSELoss)
        out_result_list.append(out_result)
        dec_result, dec_count = iTransformer_test(model_dec, CustomiTransformerDatasetCp, price_dict, seqlen=seqlen, predlen=pred_len, batch_size=batch_size, name='iTransformer_multi_dec', loss_func=criterion_MSELoss)
        dec_result_list.append(dec_result)
        print(f'模型_out 在 {out_count} 只股票上表现更优')
        print(f'模型_dec 在 {dec_count} 只股票上表现更优')
        
        # 计算每只股票的平均损失
        avg_out = {}
        avg_dec = {}
        for result in out_result_list:
            for stock, loss in result.items():
                avg_out[stock] = avg_out.get(stock, 0) + loss
        for stock in avg_out:
            avg_out[stock] /= len(out_result_list)

        for result in dec_result_list:
            for stock, loss in result.items():
                avg_dec[stock] = avg_dec.get(stock, 0) + loss
        for stock in avg_dec:
            avg_dec[stock] /= len(dec_result_list)

        # 比较两个模型的平均损失，并统计表现更优的股票数量
        better_out = 0
        better_dec = 0
        for stock in avg_out:
            if avg_out[stock] < avg_dec[stock]:
                better_out += 1
                print(f'{stock}: model_out 更优')
            else:
                better_dec += 1
                print(f'{stock}: model_dec 更优')
        
        print(f'模型_out 在 {better_out} 只股票上表现更优')
        print(f'模型_dec 在 {better_dec} 只股票上表现更优')
