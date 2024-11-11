import os
import pandas as pd

import config
from preprocessors import FeatureEngineer, data_split

from StockEnv.StockEnv import StockTradingEnv
from StockEnv.config import INDICATORS as INDICATORS

stock_nums = ['000001', '600031',]
index = ['日期_Date', '股票代码_Stkcd', '收盘价_Clpr', '开盘价_Oppr', '最高价_Hipr', '最低价_Lopr', '复权价1(元)_AdjClpr1',
         '复权价2(元)_AdjClpr2', '成交量_Trdvol',
         '成交金额_Trdsum', '日振幅(%)_Dampltd', '总股数日换手率(%)_DFulTurnR', '流通股日换手率(%)_DTrdTurnR',
         '日收益率_Dret', '日资本收益率_Daret',
         '等权平均市场日收益率_Dreteq', '流通市值加权平均市场日收益率_Drettmv', '总市值加权平均市场日收益率_Dretmc',
         '等权平均市场日资本收益率_Dareteq',
         '总市值加权平均日资本收益_Daretmc', '日无风险收益率_DRfRet', '市盈率_PE']

df = pd.DataFrame()
type_dict = {'股票代码_Stkcd': 'str'}

for stock_num in stock_nums:
    file_path = '/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/股票数据/' + stock_num + '/'
    file_names = os.listdir(file_path)

    for file_name in file_names:
        xls_path = file_path + file_name
        sheet_name = pd.ExcelFile(xls_path).sheet_names[0]
        # 确认列索引是否在范围内
        new_df = pd.read_excel(xls_path, sheet_name=sheet_name, header=0, dtype=type_dict)
        df = pd.concat([df, new_df])

df = df[index]
df = df.sort_values(['日期_Date','股票代码_Stkcd'],ignore_index=True)

# 重命名列名以符合stockstats要求
df.rename(columns={
    '日期_Date': 'date',
    '股票代码_Stkcd': 'tic',
    '收盘价_Clpr': 'close',
    '开盘价_Oppr': 'open',
    '最高价_Hipr': 'high',
    '最低价_Lopr': 'low',
    '成交量_Trdvol': 'volume'
}, inplace=True)

# 找到每只股票的最晚开始日期和最早结束日期
latest_start_date = df.groupby('tic')['date'].min().max()
earliest_end_date = df.groupby('tic')['date'].max().min()

# 过滤数据框以仅包含这些日期范围内的数据
df = df[(df['date'] >= latest_start_date) & (df['date'] <= earliest_end_date)]

# 确保每只股票包含相同日期的交易数据
all_dates = pd.date_range(start=latest_start_date, end=earliest_end_date)
df = df.pivot_table(index='date', columns='tic', values=df.columns.difference(['date', 'tic']))
df = df.reindex(all_dates).fillna(method='ffill').fillna(method='bfill')
df = df.stack(level='tic').reset_index()

# 重命名列名
df.rename(columns={'level_0': 'date'}, inplace=True)
# 打印重命名后的数据框

fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
                    use_turbulence=True,
                    user_defined_feature = False)

df = fe.preprocess_data(df)
df = data_split(df, '2016-01-04', '2024-06-28')

stock_dimension = len(stock_nums)

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension
state_space = 1 + 3 * stock_dimension + len(INDICATORS) * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,

    'seq_len':12,
    'pred_len':1,
    'nhead':8,
    'nlayers':6,
    'd_ff':2048,
    'keys': stock_nums,
    'dmodel':256
}

env = StockTradingEnv(df, **env_kwargs)

