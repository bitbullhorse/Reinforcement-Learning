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
from funcutil import *

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    dmodel = 128
    seqlen=12
    batch_size = 32
    EPOCHS = 100
    start_len = 5
    end_len = 11

    stock_nums = config.STOCK_NUMS
    data_path = config.DATA_PATH

    for stock_num in stock_nums:
        # if config.is_stock_trained(stock_num):
        #     print(f"Stock {stock_num} has already been trained.")
        #     continue

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
            model = TransformerCp(d_model=20, nhead=4,pred_len=pred_len).to(device)
            train_transformer_cp(EPOCHS, model, CustomDatasetCp, stock_num, price, seqlen=seqlen, batch_size=batch_size,
                                 name='cp', predlen=pred_len, loss_func=criterion_MSELoss)
            lstm = PureLSTMRegression(input_size=20, hidden_size=128, dropout=0.1, pred_len=pred_len).to(device)
            train_itransformer_cp(EPOCHS, lstm, CustomiTransformerDatasetCp, stock_num, price, seqlen=seqlen, predlen=pred_len, batch_size=batch_size, name='cp', loss_func=criterion_MSELoss, model_name='LSTM')
        # config.add_trained_stock(stock_num)