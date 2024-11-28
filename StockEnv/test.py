import os
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from DLinear import Model as DLinear
from PatchTST import Model as PatchTST

import config
from iTransformer_model import multi_iTransformer
from DLinear import iTransformer_multi_Dlinear

from datautil import CustomiTransformerDatasetCp
from train_mo_iTransformer import price_split
index = config.INDEX

criterion_CrossEntropy = nn.CrossEntropyLoss()
criterion_NLLLoss = nn.NLLLoss()
criterion_PoissonNLLLoss = nn.PoissonNLLLoss()
criterion_L1Loss = nn.L1Loss()
criterion_MSELoss = nn.MSELoss()

seq_len = 36
pred_lens = [5, 10]
dmodel = 256
stock_nums = os.listdir(config.DATA_PATH)
data_path = config.DATA_PATH

multi_cp_path = '/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/multi_cp/'
saved_model_path = '/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/saved_model/'

price_dict = {}

for stock_name in stock_nums:
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

for pred_len in pred_lens:
    dlinear = DLinear(enc_in=len(index) - 1, seq_len=seq_len, pred_len=pred_len, individual=0)
    dlinear.to(config.DEVICE)
    dlinear.double()
    dlinear.eval()

    patchTST = PatchTST(cin=len(index) - 1, dmodel=128, nheads=8, nlayers=6, seq_len=seq_len, d_ff=256, pred_len=pred_len)
    patchTST.eval()
    patchTST.to(config.DEVICE)

    multi_out = multi_iTransformer(12, pred_len, 256, 8, 6, 4096, stock_nums)
    multi_out.eval()
    multi_out.load_state_dict(torch.load(multi_cp_path + 'iTransformer_multi_out_' + str(pred_len) + '.pth'))
    multi_out.double()
    multi_out.to(config.DEVICE)

    multi_Dl = iTransformer_multi_Dlinear(d_model=dmodel, n_head=8, nlayers=6, seq_len=seq_len, d_ff=4096, pred_len=pred_len, individual=0, keys=stock_nums)
    multi_Dl.to(config.DEVICE)
    multi_Dl.eval()
    try:
        multi_Dl.load_state_dict(torch.load(multi_cp_path + 'iTransformer_multi_Dlinear_' + str(pred_len) + '.pth'))
        multi_Dl.double()
    except:
        print('No model found.')
        continue
    count = 0
    with torch.no_grad():
        for stock_name, price in tqdm(price_dict.items(), desc='Testing'):
            dlinear.load_state_dict(torch.load(saved_model_path + stock_name + '/cp/DLinear_' + str(pred_len) + '.pth'))
            dlinear.double()
            try:
                patchTST.load_state_dict(torch.load(saved_model_path + stock_name + '/cp/PatchTST_' + str(pred_len) + '.pth'))
                patchTST.double()
            except:
                print(saved_model_path + stock_name + '/cp/PatchTST_' + str(pred_len) + '.pth')
                continue
            _, _, test_price = price_split(price)
            label = []
            pred = []
            test_data = CustomiTransformerDatasetCp(test_price, seq_len, pred_len)
            test_data_12 = CustomiTransformerDatasetCp(test_price, 12, pred_len)
            
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
            test_loader_12 = torch.utils.data.DataLoader(test_data_12, batch_size=1, shuffle=False)

            loss_dlinear = 0
            loss_patchTST = 0
            loss_multi_out = 0
            loss_multi_Dl = 0
            
            for inputs, labels in test_loader:
                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                
                output_dlinear = dlinear(inputs)
                output_patchTST = patchTST(inputs)
                output_multi_Dl = multi_Dl(inputs, stock_name)

                loss_dlinear += criterion_MSELoss(output_dlinear[:,-pred_len,0], labels[:, -pred_len, 0]).item()
                loss_patchTST += criterion_MSELoss(output_patchTST[:,-pred_len,0], labels[:, -pred_len, 0]).item()
                loss_multi_Dl += criterion_MSELoss(output_multi_Dl[:,-pred_len,0], labels[:, -pred_len, 0]).item()
            
            for inputs, labels in test_loader_12:
                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                
                output_multi_out = multi_out(inputs, stock_name)

                loss_multi_out += criterion_MSELoss(output_multi_out[:,-pred_len,0], labels[:,-pred_len,0]).item()
            
            print(f'{stock_name} {pred_len} dlinear loss: {loss_dlinear / len(test_loader)}')
            print(f'{stock_name} {pred_len} patchTST loss: {loss_patchTST / len(test_loader)}')
            print(f'{stock_name} {pred_len} multi_out loss: {loss_multi_out / len(test_loader_12)}')
            print(f'{stock_name} {pred_len} multi_Dl loss: {loss_multi_Dl / len(test_loader)}')
            if loss_patchTST / len(test_loader) > loss_multi_out / len(test_loader_12):
                count += 1
    print(f'{pred_len} count: {count}')
