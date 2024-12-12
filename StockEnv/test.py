import os
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from DLinear import Model as DLinear
from PatchTST import Model as PatchTST

import config
from iTransformer_model import multi_iTransformer, iTransformer
from DLinear import iTransformer_multi_Dlinear

from datautil import CustomiTransformerDatasetCp
from train_mo_iTransformer import price_split
index = config.INDEX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

stock10 = ['600030', '600519', '601688', '601901', '601288', '600010', '600050', '603288', '601919', '000001',]
stock20 = ['600030', '600519', '601688', '601901', '601288', '600010', '600050', '603288', '601919', '000001',
            '601601', '600585', '600016', '601398', '601318', '600104', '600000', '600028', '600372', '600809',]



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

    dlinear_12 = DLinear(enc_in=len(index) - 1, seq_len=12, pred_len=pred_len, individual=0)
    dlinear_12.to(config.DEVICE)
    dlinear_12.double()
    dlinear_12.eval()

    patchTST = PatchTST(cin=len(index) - 1, dmodel=128, nheads=8, nlayers=6, seq_len=seq_len, d_ff=256, pred_len=pred_len, patch_len=12, stride=8)
    patchTST.eval()
    patchTST.to(config.DEVICE)

    patchTST_12 = PatchTST(cin=len(index) - 1, dmodel=128, nheads=8, nlayers=6, seq_len=12, d_ff=256, pred_len=pred_len, patch_len=4, stride=3)
    patchTST_12.eval()
    patchTST_12.to(config.DEVICE)

    itransformer = iTransformer(12, pred_len, 128, 8, 6, 512,)
    itransformer.eval()
    itransformer.to(config.DEVICE)

    multi_out = multi_iTransformer(12, pred_len, 256, 8, 6, 4096, stock_nums)
    multi_out.eval()
    multi_out.load_state_dict(torch.load(multi_cp_path + 'iTransformer_multi_out_' + str(pred_len) + '.pth', weights_only=True))
    multi_out.double()
    multi_out.to(config.DEVICE)

    multi_36_out = multi_iTransformer(36, pred_len, 256, 8, 6, 4096, stock_nums)
    multi_36_out.eval()
    multi_36_out.load_state_dict(torch.load(multi_cp_path + 'iTransformer_multi_36_out_' + str(pred_len) + '.pth', weights_only=True))
    multi_36_out.double()
    multi_36_out.to(config.DEVICE)

    multi_Dl = iTransformer_multi_Dlinear(d_model=dmodel, n_head=8, nlayers=6, seq_len=seq_len, d_ff=4096, pred_len=pred_len, individual=0, keys=stock_nums)
    multi_Dl.to(config.DEVICE)
    multi_Dl.eval()

    new_multi_out10 = multi_iTransformer(12, pred_len, 256, 8, 6, 4096, stock10)
    new_multi_out10.eval()
    new_multi_out10.double()
    try:
        new_multi_out10.load_state_dict(torch.load(multi_cp_path + f'new_iTransformer_{len(stock10)}_{12}_out_{pred_len}.pth', weights_only=True))
    except:
        pass
    new_multi_out10.to(config.DEVICE)

    new_multi_out20 = multi_iTransformer(12, pred_len, 256, 8, 6, 4096, stock20)
    new_multi_out20.eval()
    new_multi_out20.double()
    try:
        new_multi_out20.load_state_dict(torch.load(multi_cp_path + f'new_iTransformer_{len(stock20)}_{12}_out_{pred_len}.pth', weights_only=True))
    except:
        pass
    new_multi_out20.to(config.DEVICE)

    fixed_multi_out = multi_iTransformer(12, pred_len, 256, 8, 6, 4096, stock_nums)
    fixed_multi_out.eval()
    fixed_multi_out.double()
    fixed_multi_out.to(config.DEVICE)
    try:
        fixed_multi_out.load_state_dict(torch.load(multi_cp_path + f'fixed_iTransformer_{len(stock_nums)}_{12}_out_{pred_len}.pth', weights_only=True))
    except:
        pass

    try:
        multi_Dl.load_state_dict(torch.load(multi_cp_path + 'iTransformer_multi_Dlinear_' + str(pred_len) + '.pth', weights_only=True))
        multi_Dl.double()
    except:
        print('No model found.')
        continue
    count = 0
    total = 0
    count10 = 0
    count20 = 0
    count20_10 = 0
    count10_1 = 0
    with torch.no_grad():
        for stock_name, price in tqdm(price_dict.items(), desc='Testing'):
            dlinear.load_state_dict(torch.load(saved_model_path + stock_name + '/cp/DLinear_' + str(pred_len) + '.pth', weights_only=True))
            dlinear.double()

            dlinear_12.load_state_dict(torch.load(saved_model_path + stock_name + '/cp/DLinear_12_' + str(pred_len) + '.pth', weights_only=True))
            dlinear_12.double()

            itransformer.load_state_dict(torch.load(saved_model_path + stock_name + '/cp/iTransformer_' + str(pred_len) + '.pth', weights_only=True))
            itransformer.double()
            try:
                patchTST.load_state_dict(torch.load(saved_model_path + stock_name + '/cp/PatchTST_' + str(pred_len) + '.pth', weights_only=True))
                patchTST.double()
                patchTST_12.load_state_dict(torch.load(saved_model_path + stock_name + '/cp/PatchTST_12_' + str(pred_len) + '.pth', weights_only=True))
                patchTST_12.double()
            except:
                print(saved_model_path + stock_name + '/cp/PatchTST_' + str(pred_len) + '.pth')
                continue
            total += 1
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
            loss_itr = 0
            loss_multi_36_out = 0
            loss_dlinear_12 = 0
            loss_patchTST_12 = 0
            loss_multi_10_out = 0
            loss_multi_20_out = 0
            loss_multi_fixed_out = 0
            
            for inputs, labels in test_loader:
                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                
                output_dlinear = dlinear(inputs)
                output_patchTST = patchTST(inputs)
                output_multi_Dl = multi_Dl(inputs, stock_name)
                output_multi_36_out = multi_36_out(inputs, stock_name)

                loss_dlinear += criterion_MSELoss(output_dlinear[:,:,0], labels[:,:, 0]).item()
                loss_patchTST += criterion_MSELoss(output_patchTST[:,:,0], labels[:, :, 0]).item()
                loss_multi_Dl += criterion_MSELoss(output_multi_Dl[:,:,0], labels[:, :, 0]).item()
                loss_multi_36_out += criterion_MSELoss(output_multi_36_out[:,:,0], labels[:, :, 0]).item()

            for inputs, labels in test_loader_12:
                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                
                output_multi_out = multi_out(inputs, stock_name)
                outout_itr = itransformer(inputs)
                output_fixed_out = fixed_multi_out(inputs, stock_name)
                output_dlinear_12 = dlinear_12(inputs)
                output_patchTST_12 = patchTST_12(inputs)
                try:
                    output_multi_10_out = new_multi_out10(inputs, stock_name)
                    loss_multi_10_out += criterion_MSELoss(output_multi_10_out[:,:,0], labels[:,:,0]).item()
                except:
                    loss_multi_10_out = torch.inf
                
                try:
                    output_multi_20_out = new_multi_out20(inputs, stock_name)
                    loss_multi_20_out += criterion_MSELoss(output_multi_20_out[:,:,0], labels[:,:,0]).item()
                except:
                    loss_multi_20_out = torch.inf

                loss_multi_out += criterion_MSELoss(output_multi_out[:,:,0], labels[:,:,0]).item()
                loss_itr += criterion_MSELoss(outout_itr[:,:,0], labels[:,:,0]).item()
                loss_dlinear_12 += criterion_MSELoss(output_dlinear_12[:,:,0], labels[:,:,0]).item()
                loss_patchTST_12 += criterion_MSELoss(output_patchTST_12[:,:,0], labels[:,:,0]).item()
                loss_multi_fixed_out += criterion_MSELoss(output_fixed_out[:,:,0], labels[:,:,0]).item()

            tmp = [
                    loss_patchTST / len(test_loader), 
                    loss_itr / len(test_loader_12), 
                    loss_multi_Dl / len(test_loader),
                    loss_multi_36_out / len(test_loader),
                    loss_patchTST_12 / len(test_loader_12),
                    loss_multi_fixed_out / len(test_loader_12),
                   ]
            if loss_multi_out / len(test_loader_12) < min(tmp):
                count += 1
            if loss_multi_out < loss_multi_20_out and loss_multi_20_out != torch.inf:
                print(f'{stock_name} loss_multi_out:{loss_multi_out}, loss_multi_20_out:{loss_multi_20_out}')
                count20 += 1
            if loss_multi_out < loss_multi_10_out and loss_multi_10_out != torch.inf:
                print(f'{stock_name} loss_multi_out:{loss_multi_out}, loss_multi_10_out:{loss_multi_10_out}')
                count10 += 1
            if loss_multi_20_out < loss_multi_10_out and loss_multi_10_out != torch.inf and loss_multi_20_out != torch.inf:
                count20_10 += 1
            if loss_multi_10_out < loss_itr:
                count10_1 += 1
    print(f'pred_len:{pred_len}, total:{total},count:{count},count / total: {count / total},count20:{count20}, count10:{count10}, count20_10:{count20_10}, count10_1:{count10_1}')
