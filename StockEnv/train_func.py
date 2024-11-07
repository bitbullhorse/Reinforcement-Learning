import os
from matplotlib import pyplot as plt
from numpy import inf
from torch.utils.data import DataLoader as Dataloader
# from torch.utils.data.dataloader import Dataloader
import torch
import torch.nn as nn
import os
from datetime import date
import time  # 添加导入
from tqdm import tqdm  # 添加导入

device = 'cuda'
cwd = '/home/czj/pycharm_project_tmp_pytorch/VaR/'
torch.autograd.set_detect_anomaly(True)

criterion_CrossEntropy = nn.CrossEntropyLoss()
criterion_NLLLoss = nn.NLLLoss()
criterion_PoissonNLLLoss = nn.PoissonNLLLoss()
criterion_L1Loss = nn.L1Loss()
criterion_MSELoss = nn.MSELoss()

def normalize_list(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

def log_training_info(path, model_name, end_time, loss_sum, loss_func, predlen=None, model=None):
    log_path = os.path.join(path, f'{model_name}_log.txt')
    end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))  # 转换时间格式
    with open(log_path, 'a') as log_file:
        if predlen is None:
            log_file.write(f'Model: {model_name}, settings:dmodel:{model.d_model}, nhead:{model.n_head} nlayers:{model.nlayers} seqlen:{model.seq_len} d_ff:{model.d_ff}\
                           \n        End Time: {end_time_str}, Loss: {loss_sum}, Loss_func: {(str(type(loss_func))[8:-2]).split(".")[-1]}\n')
        else:
            model_setting = model.__dict__
            tmp = {}
            for k, v in model_setting.items():
                if isinstance(v, int):
                    tmp[k] = v
            log_file.write(f'Model: {model_name}_{predlen}, {tmp}\n\
         End Time: {end_time_str}, Loss: {loss_sum}, Loss_func: {(str(type(loss_func))[8:-2]).split(".")[-1]}\n')

def train_transformer(model, epochs, train_loader:Dataloader, Loss, optimizer, scheduler, eval_loader:Dataloader, \
                   test_loader:Dataloader, stock_num:str, name:str, pred_len:int=1):
    start_time = time.time()  # 记录开始时间
    model.to(device)
    model.train()
    count = 0
    loss_list = []
    test_close_list = []
    close_list = []
    minloss = inf
    path = '/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/saved_model/' + stock_num + '/' + name
    os.makedirs(path, exist_ok=True)
    epoch = 0
    saved_flag = False
    try:
        for epoch in range(epochs):
            print(f'*********pred_len:{pred_len},epoch:{epoch},model:{str(type(model))[20:-2]}**********')
            for src, tgt, cp in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch'):
                optimizer.zero_grad()
                count+=1
                output = model(src, src, has_mask=False)
                loss = Loss(output, cp)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            print(f'**********************************************************')
            scheduler.step()
            loss_sum = 0
            for src, tgt, cp in eval_loader:
                output = model(src, src, has_mask=False)
                loss = Loss(output, cp)
                loss_sum+=loss.item()
            if loss_sum < minloss:
                minloss = loss_sum
                saved_flag = True
                torch.save(model.state_dict(), path + '/' + str(type(model))[20:-2] + f'_{pred_len}.pth')
            print('Eval Loss:', loss_sum)
            loss_list.append(loss_sum)
    except KeyboardInterrupt:
        print('Training interrupted by user.')
    if not saved_flag:
        print('No model saved.')
        return
    if epoch >= 2:
        loss_list = normalize_list(loss_list)
    model.load_state_dict(torch.load(path + '/' + str(type(model))[20:-2] + f'_{pred_len}.pth'))
    print(path + '/' + str(type(model))[20:-2] + f'_{pred_len}.pth')
    day = 0
    loss_sum = 0
    for src, _, cp in test_loader:
        day+=test_loader.batch_size
        output = model(src, src, has_mask=False)
        close_list+=cp.view(-1).tolist()
        test_close_list+=output.view(-1).tolist()
        loss_sum+=Loss(output, cp).item()
    print('Test Loss:', loss_sum)
    current_date = date.today()
    fig, axes = plt.subplots(2, 1)
    x_label = list(range(len(loss_list)))
    axes[0].plot(loss_list)
    axes[0].set_title('Loss function',)
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    
    x1_label = list(range(len(test_close_list)))
    axes[1].plot(test_close_list, 'r-', label=u'predicted price')
    axes[1].plot(close_list, 'b-', label=u'close price')
    axes[1].set_title('Close Price')
    axes[1].set_xlabel('day')
    axes[1].set_ylabel('price')
    plt.legend(loc='upper left', fontsize='large')
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(path + '/' + str(type(model))[20:-2] + f'_loss&close_{pred_len}.png')
    plt.show()
    
    end_time = time.time()  # 记录结束时间
    print(f'Training time: {end_time - start_time} seconds')  # 打印训练时间
    log_training_info(path, str(type(model))[20:-2], end_time, loss_sum, Loss, pred_len, model=model)  # 调整参数顺序

def train_iTranformer(model, epochs, train_loader:Dataloader, test_Loss, optimizer, scheduler, eval_loader:Dataloader, \
                      test_loader:Dataloader, stock_num:str = None, name:str=None):
    start_time = time.time()  # 记录开始时间
    model.train()
    model.to(device)
    count = 0
    loss_list = []
    minloss = inf
    path = '/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/saved_model/' + stock_num + '/' + name
    os.makedirs(path, exist_ok=True)
    epoch = 0
    saved_flag = False
    try:
        for epoch in range(epochs):
            print(f'*******predlen:{model.pred_len}, epoch:{epoch},model:iTransformer_{model.pred_len}******')
            for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch'):
                optimizer.zero_grad()
                count += 1
                output = model(x)
                loss = criterion_MSELoss(output[:,:,0], y[:,:,0])
                loss.backward()
                optimizer.step()
            print(f'***************************************************')

            scheduler.step()
            loss_sum = 0
            for x, y in eval_loader:
                output = model(x)
                loss = criterion_MSELoss(y[:,:,0], output[:,:,0])
                loss_sum += loss.item()
            print('Eval Loss:', loss_sum)
            if loss_sum < minloss:
                minloss = loss_sum
                torch.save(model.state_dict(), path + '/' + f'iTransformer_{model.pred_len}.pth')
                saved_flag = True
            loss_list.append(loss_sum)
    except KeyboardInterrupt:
        print('Training interrupted by user.')
    if not saved_flag:
        print('No model saved.')
        return
    label = []
    pred = []
    if epoch >= 2:
        loss_list = normalize_list(loss_list)
    model.load_state_dict(torch.load(path + '/' + f'iTransformer_{model.pred_len}.pth'))
    model.eval()
    loss_sum = 0
    for x, y in test_loader:
        output = model(x)
        loss = test_Loss(y[:, :, 0], output[:, :, 0])
        output = output[:, -1, 0].reshape(-1).tolist()  # 使用 reshape
        y = y[:, -1, 0].reshape(-1).tolist()  # 使用 reshape
        loss_sum += loss.item()
        label+=y
        pred+=output
    print('Test Loss:', loss_sum)
    current_date = date.today()
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(loss_list)
    axes[0].set_title('Loss function', )
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')

    axes[1].plot(label, 'b-', label='label')
    axes[1].plot(pred, 'r-', label='predict')
    axes[1].set_title('Close Price')
    axes[1].set_xlabel('day')
    axes[1].set_ylabel('price')
    plt.legend()
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(path + '/' + f'iTransformer_loss&close_{model.pred_len}.png')
    plt.show()
    
    end_time = time.time()  # 记录结束时间
    print(f'Training time: {end_time - start_time} seconds')  # 打印训练时间
    log_training_info(path, f'iTransformer_{model.pred_len}', end_time, loss_sum, test_Loss)  # 调整参数顺序
