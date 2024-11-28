import os
from DLinear import Model as DLinearModel

from datautil import CustomiTransformerDatasetCp

from funcutil import *

if __name__ == '__main__':
    seqlen = 36
    batch_size = 16
    EPOCHS = 200

    stock_nums = config.STOCK_NUMS
    data_path = config.DATA_PATH

    for stock_num in stock_nums:
        file_path = data_path + stock_num
        file_names = os.listdir(file_path)
        price = pd.DataFrame()
        for file_name in file_names:
            xls_path = file_path + file_name
            sheet_name = pd.ExcelFile(xls_path).sheet_names[0]
            # 确认列索引是否在范围内
            new_price = pd.read_excel(xls_path, sheet_name=sheet_name, header=0)
            price = pd.concat([price, new_price])

        pred_lens = [1, 5, 10]
        for pred_len in pred_lens:
            model = DLinearModel(enc_in=len(index) - 1, seq_len=seqlen, pred_len=pred_len, individual=0)
            model.double()
            train_itransformer_cp(EPOCHS, model, CustomiTransformerDatasetCp, stock_num, price, seqlen, pred_len,
                                  batch_size, name='cp', loss_func=criterion_L1Loss, model_name='DLinear')
