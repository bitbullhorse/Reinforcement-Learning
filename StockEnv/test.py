import os
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from DLinear import Model as DLinear
from PatchTST import Model as PatchTST
from Transformer import PureLSTMRegression, TransformerCp
import config
from iTransformer_model import multi_iTransformer, iTransformer
from DLinear import iTransformer_multi_Dlinear
import openpyxl
import matplotlib.pyplot as plt

from datautil import CustomiTransformerDatasetCp
from train_mo_iTransformer import price_split
index = config.INDEX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion_CrossEntropy = nn.CrossEntropyLoss()
criterion_NLLLoss = nn.NLLLoss()
criterion_PoissonNLLLoss = nn.PoissonNLLLoss()
criterion_L1Loss = nn.L1Loss()
criterion_MSELoss = nn.MSELoss()

loss_func = criterion_MSELoss

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
    test_len = pred_len
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

    transformer = TransformerCp(d_model=len(index) - 1, nhead=4, pred_len=pred_len,)
    transformer.eval()
    transformer.to(config.DEVICE)

    lstm = PureLSTMRegression(input_size=len(index) - 1, hidden_size=128, dropout=0.1, pred_len=pred_len).to(device)
    lstm.eval()
    lstm.double()

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

    multi_dff512_out = multi_iTransformer(12, pred_len, 256, 8, 6, 512, stock_nums)
    multi_dff512_out.eval()
    multi_dff512_out.double()
    multi_dff512_out.to(config.DEVICE)
    multi_dff512_out.load_state_dict(torch.load(multi_cp_path + f'dff512_iTransformer_{len(stock_nums)}_{12}_out_{pred_len}.pth', weights_only=True))

    try:
        multi_Dl.load_state_dict(torch.load(multi_cp_path + 'iTransformer_multi_Dlinear_' + str(pred_len) + '.pth', weights_only=True))
        multi_Dl.double()
    except:
        print('No model found.')
        continue

    count = 0 # multi_out < min
    total = 0
    count10 = 0 # multi_out < iTransformer
    count20 = 0 # multi_out < multi_20_out
    count20_10 = 0 # multi_20_out < multi_10_out
    count10_1 = 0 # multi_10_out < iTransformer
    count_fixed = 0 # fixed_multi_out best
    count_random_fixed = 0 # multi_out < fixed_multi_out
    countdl = 0 # iTransformer_multi_out < DLinear
    countimulti_dl = 0 # iTransformer_multi_Dlinear < DLinear
    count_multi_patchTST_36 = 0
    count_multi_dl_36 = 0
    count_multi_patchTST_12 = 0
    count20_1 = 0

    best_model_counts = {
        'patchTST': 0,
        'patchTST_12': 0,
        'itransformer': 0,
        'transformer': 0,
        'lstm': 0,
        'dlinear': 0,
        'dlinear_12': 0,
        'multi_out': 0
    }

    # pred_len:10, total:32,count:18,random order better than fixed order:24,
    # multi-iTr win: 0.5625,
    # fixed multi-iTr win:14, 
    # multi-iTr better than 20-multi-iTr:16, 
    # multi-iTr better than 10-multi-iTr:7, 
    # 20-multi-iTr better than 10-multi-iTr:6, 
    # 10-multi-iTr better than itransformer:9 
    # multi-iTr better than patchTST_36:29,
    # multi-iTr better than dl_36:22
    # multi-iTr better than patchTST_12:22
    # multi-iTr better than dl_12:32

    # pred_len:5, total:32,count:24,random order better than fixed order:30,
    # multi-iTr win: 0.75,
    # fixed multi-iTr win:21, 
    # multi-iTr better than 20-multi-iTr:18, 
    # multi-iTr better than 10-multi-iTr:8, 
    # 20-multi-iTr better than 10-multi-iTr:3, 
    # 10-multi-iTr better than itransformer:10 
    # multi-iTr better than patchTST_36:32,
    # multi-iTr better than dl_36:21
    # multi-iTr better than patchTST_12:26
    # multi-iTr better than dl_12:31

    results = {
        'stock_name': [],
        'dlinear': [],
        'patchTST': [],
        'transformer': [],
        'lstm': [],
        'multi_out': [],
        'multi_Dl': [],
        'itransformer': [],
        'multi_36_out': [],
        'dlinear_12': [],
        'patchTST_12': [],
        'multi_10_out': [],
        'multi_20_out': [],
        'fixed_multi_out': [],
        'multi_dff512_out': []
    }

    with torch.no_grad():
        for stock_name, price in tqdm(price_dict.items(), desc='Testing'):
            dlinear.load_state_dict(torch.load(saved_model_path + stock_name + '/cp/DLinear_' + str(pred_len) + '.pth', weights_only=True))
            dlinear.double()

            dlinear_12.load_state_dict(torch.load(saved_model_path + stock_name + '/cp/DLinear_12_' + str(pred_len) + '.pth', weights_only=True))
            dlinear_12.double()

            try:
                transformer.load_state_dict(torch.load(saved_model_path + stock_name + '/cp/TransformerCp_' + str(pred_len) + '.pth', weights_only=True))
                transformer.double()
            except:
                pass

            try:
                lstm.load_state_dict(torch.load(saved_model_path + stock_name + '/cp/LSTM_' + str(pred_len) + '.pth', weights_only=True))
                lstm.double()
            except:
                pass
        
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
                
            ground_truth = [] #
            pred_multi_out = [] #
            pred_multi_36_out = []
            pred_multi_Dl = []
            pred_multi_fixed_out = []
            pred_dlinear = []
            pred_patchTST = []
            pred_transformer = [] #
            pred_lstm = [] #
            pred_itr = [] #
            pred_dlinear_12 = [] #
            pred_patchTST_12 = [] #
            
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
            loss_transformer = 0
            loss_lstm = 0
            loss_multi_out = 0
            loss_multi_Dl = 0
            loss_itr = 0
            loss_multi_36_out = 0
            loss_dlinear_12 = 0
            loss_patchTST_12 = 0
            loss_multi_10_out = 0
            loss_multi_20_out = 0
            loss_multi_fixed_out = 0
            loss_multi_dff512_out = 0
            

            for inputs, labels in test_loader:
                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                
                output_dlinear = dlinear(inputs)
                output_patchTST = patchTST(inputs)
                output_multi_Dl = multi_Dl(inputs, stock_name)
                output_multi_36_out = multi_36_out(inputs, stock_name)

                loss_dlinear += loss_func(output_dlinear[:,0:test_len,0], labels[:,0:test_len, 0]).item()
                loss_patchTST += loss_func(output_patchTST[:,0:test_len,0], labels[:, 0:test_len, 0]).item()
                loss_multi_Dl += loss_func(output_multi_Dl[:,0:test_len,0], labels[:, 0:test_len, 0]).item()
                loss_multi_36_out += loss_func(output_multi_36_out[:,0:test_len,0], labels[:, 0:test_len, 0]).item()

            i = 0
            for inputs, labels in test_loader_12:
                i += 1
                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                
                output_multi_out = multi_out(inputs, stock_name)
                outout_itr = itransformer(inputs)
                output_fixed_out = fixed_multi_out(inputs, stock_name)
                output_dlinear_12 = dlinear_12(inputs)
                output_patchTST_12 = patchTST_12(inputs)
                output_transformer = transformer(inputs, inputs)
                output_lstm = lstm(inputs)
                try:
                    output_multi_10_out = new_multi_out10(inputs, stock_name)
                    loss_multi_10_out += loss_func(output_multi_10_out[:,0:test_len,0], labels[:,0:test_len,0]).item()
                except:
                    loss_multi_10_out = torch.inf
                
                try:
                    output_multi_20_out = new_multi_out20(inputs, stock_name)
                    loss_multi_20_out += loss_func(output_multi_20_out[:,0:test_len,0], labels[:,0:test_len,0]).item()
                except:
                    loss_multi_20_out = torch.inf

                loss_multi_out += loss_func(output_multi_out[:,0:test_len,0], labels[:,0:test_len,0]).item()
                loss_itr += loss_func(outout_itr[:,0:test_len,0], labels[:,0:test_len,0]).item()
                loss_dlinear_12 += loss_func(output_dlinear_12[:,0:test_len,0], labels[:,0:test_len,0]).item()
                loss_patchTST_12 += loss_func(output_patchTST_12[:,0:test_len,0], labels[:,0:test_len,0]).item()
                loss_multi_fixed_out += loss_func(output_fixed_out[:,0:test_len,0], labels[:,0:test_len,0]).item()
                loss_transformer += loss_func(output_transformer[:,0:test_len,0], labels[:,0:test_len,0]).item()
                loss_lstm += loss_func(output_lstm[:,0:test_len,0], labels[:,0:test_len,0]).item()
                loss_multi_dff512_out += loss_func(multi_dff512_out(inputs, stock_name)[:,0:test_len,0], labels[:,0:test_len,0]).item()

                if i % pred_len == 0:
                    pred_patchTST_12+=output_patchTST_12[:,0:test_len,0].view(-1).tolist()
                    pred_transformer+=output_transformer[:,0:test_len,0].view(-1).tolist()
                    pred_lstm+=output_lstm[:,0:test_len,0].view(-1).tolist()
                    pred_itr+=outout_itr[:,0:test_len,0].view(-1).tolist()
                    pred_dlinear_12+=output_dlinear_12[:,0:test_len,0].view(-1).tolist()
                ground_truth+=labels[:,0,0].view(-1).tolist()
                pred_multi_out+=output_multi_out[:,0,0].view(-1).tolist()
                

            list_tmp = [pred_dlinear_12,pred_multi_out, pred_patchTST_12, pred_transformer, pred_lstm, pred_itr, ground_truth]

            

            results['stock_name'].append(stock_name)
            results['dlinear'].append(loss_dlinear / len(test_loader))
            results['patchTST'].append(loss_patchTST / len(test_loader))
            results['transformer'].append(loss_transformer / len(test_loader_12))
            results['lstm'].append(loss_lstm / len(test_loader_12))
            results['multi_out'].append(loss_multi_out / len(test_loader_12))
            results['multi_Dl'].append(loss_multi_Dl / len(test_loader))
            results['itransformer'].append(loss_itr / len(test_loader_12))
            results['multi_36_out'].append(loss_multi_36_out / len(test_loader))
            results['dlinear_12'].append(loss_dlinear_12 / len(test_loader_12))
            results['patchTST_12'].append(loss_patchTST_12 / len(test_loader_12))
            results['multi_10_out'].append(loss_multi_10_out / len(test_loader_12))
            results['multi_20_out'].append(loss_multi_20_out / len(test_loader_12))
            results['fixed_multi_out'].append(loss_multi_fixed_out / len(test_loader_12))
            results['multi_dff512_out'].append(loss_multi_dff512_out / len(test_loader_12))

            tmp = [
                    loss_patchTST / len(test_loader), 
                    # loss_patchTST_12 / len(test_loader_12),
                    loss_itr / len(test_loader_12), 
                    loss_transformer / len(test_loader_12),
                    loss_lstm / len(test_loader_12),
                    loss_dlinear / len(test_loader),
                ]
        

            if loss_multi_out / len(test_loader_12) < loss_dlinear_12 / len(test_loader_12):
                countdl += 1
            if loss_multi_Dl / len(test_loader) < loss_dlinear_12 / len(test_loader_12):
                countimulti_dl += 1
            if loss_multi_out / len(test_loader_12) < min(tmp):
                count += 1
            if loss_multi_fixed_out / len(test_loader_12) < min(tmp):
                count_fixed += 1
            if loss_multi_out < loss_multi_fixed_out:
                count_random_fixed += 1
            if loss_multi_out < loss_multi_20_out and loss_multi_20_out != torch.inf:
                count20 += 1
            if loss_multi_out < loss_multi_10_out and loss_multi_10_out != torch.inf:
                count10 += 1
            if loss_multi_20_out < loss_multi_10_out and loss_multi_10_out != torch.inf and loss_multi_20_out != torch.inf:
                count20_10 += 1
            if loss_multi_10_out < loss_itr:
                count10_1 += 1
            if loss_multi_out / len(test_loader_12) < loss_patchTST / len(test_loader):
                count_multi_patchTST_36 += 1
            if loss_multi_out / len(test_loader_12) < loss_dlinear / len(test_loader):
                count_multi_dl_36 += 1
            if loss_multi_out / len(test_loader_12) < loss_patchTST_12 / len(test_loader_12):
                count_multi_patchTST_12 += 1
            if loss_multi_20_out / len(test_loader_12) < loss_itr / len(test_loader_12):
                count20_1 += 1


            # 绘制每只股票的预测值与真实值
            plt.figure(figsize=(10, 6))
            plt.plot(ground_truth, label='Ground Truth')
            plt.plot(pred_multi_out, label='Predicted')
            plt.legend()
            plt.savefig(f'/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/figtmp/{stock_name}_{pred_len}.png')
            plt.close()

    # 创建大图
    fig, axes = plt.subplots(10, 3, figsize=(60, 120))
    # 调整子图之间的间距
    fig.subplots_adjust(hspace=0, wspace=0)
    axes = axes.flatten()

    for i, stock_name in enumerate(stock_nums[:30]):
        img = plt.imread(f'/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/figtmp/{stock_name}_{pred_len}.png')
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(stock_name, fontsize=20)  # 设置标题和标题文字大小

    plt.savefig(f'/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/fig/all_stocks_1_{pred_len}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    df = pd.DataFrame(results)
    with pd.ExcelWriter('model_losses.xlsx', engine='openpyxl', mode='a' if os.path.exists('model_losses.xlsx') else 'w', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=f'pred_len_{pred_len}', index=False)

    print(f'pred_len:{pred_len}, total:{total},count:{count},random order better than fixed order:{count_random_fixed},\n\
multi-iTr win: {count / total},\n\
fixed multi-iTr win:{count_fixed}, \n\
multi-iTr better than 20-multi-iTr:{count20}, \n\
multi-iTr better than 10-multi-iTr:{count10}, \n\
20-multi-iTr better than 10-multi-iTr:{count20_10}, \n\
20-multi-iTr better than itransformer:{count20_1}, \n\
10-multi-iTr better than itransformer:{count10_1} \n\
multi-iTr better than patchTST_36:{count_multi_patchTST_36},\n\
multi-iTr better than patchTST_12:{count_multi_patchTST_12}\n\
multi-iTr better than dl_36:{count_multi_dl_36}\
')
    print(f'multi-iTr better than dl_12:{countdl}',)
