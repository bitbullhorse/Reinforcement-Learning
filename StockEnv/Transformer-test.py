import pandas as pd
import os
from Transformer import *
from datautil import *
from train_func import train_iTranformer, train_transformer
try:
    from StockEnv import config
except:
    import config

from iTransformer_model import iTransformer

torch.autograd.set_detect_anomaly(True)

index = config.INDEX

criterion_CrossEntropy = nn.CrossEntropyLoss()
criterion_NLLLoss = nn.NLLLoss()
criterion_PoissonNLLLoss = nn.PoissonNLLLoss()
criterion_L1Loss = nn.L1Loss()
criterion_MSELoss = nn.MSELoss()

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


def train_itransformer_cp(epochs, model:nn.Module, CustomiTransformerDataset, stock_num:str, price:pd.DataFrame, seqlen=12, predlen=5, batch_size=16, name='cp', loss_func=criterion_MSELoss):
    model.to(device)

    train_price, eval_price, test_price = price_split(price)

    dataset_train = CustomiTransformerDataset(train_price, seqlen, predlen)
    dataset_eval = CustomiTransformerDataset(eval_price, seqlen, predlen)
    dataset_test = CustomiTransformerDataset(test_price, seqlen, predlen)

    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(dataset_eval, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)

    train_iTranformer(model, epochs, train_dataloader, loss_func, optimizer, scheduler, eval_dataloader,
                      test_dataloader, stock_num[0:-1], name)

def train_transformer_cp(epochs, model:nn.Module, CustomDataset, stock_num, price:pd.DataFrame, seqlen=12, batch_size=16, name='cp', predlen=1, loss_func=criterion_MSELoss,):
    model.to(device)
    train_price, eval_price, test_price = price_split(price)

    train_data = CustomDataset(train_price, seqlen, predlen)
    eval_data = CustomDataset(eval_price, seqlen, predlen)
    test_data = CustomDataset(test_price, seqlen, predlen)
    
    train_loader = None
    eval_loader = None
    test_loader = None

    if isinstance(model, TrXL):
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
        eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
    else:
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
    start_len = 5
    end_len = 11

    stock_nums = config.STOCK_NUMS
    data_path = config.DATA_PATH

    for stock_num in stock_nums:
        if config.is_stock_trained(stock_num):
            print(f"Stock {stock_num} has already been trained.")
            continue

        file_path = data_path + stock_num
        file_names = os.listdir(file_path)
        price = pd.DataFrame()
        for file_name in file_names:
            xls_path = file_path + file_name
            sheet_name = pd.ExcelFile(xls_path).sheet_names[0]
            # 确认列索引是否在范围内
            new_price = pd.read_excel(xls_path, sheet_name=sheet_name, header=0)
            price = pd.concat([price, new_price])
        pred_lens = [5,10]
        for pred_len in pred_lens:
            model = iTransformer(d_model=128, n_head=8, nlayers=6, seq_len=seqlen, d_ff=512, pred_len=pred_len)
            train_itransformer_cp(EPOCHS, model, CustomiTransformerDatasetCp, stock_num, price, seqlen, pred_len, batch_size, name='cp', loss_func=criterion_L1Loss)
        
        config.add_trained_stock(stock_num)
