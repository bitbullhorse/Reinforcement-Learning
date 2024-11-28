import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
import datetime

import config
from preprocessors import FeatureEngineer, data_split

from StockEnv.StockEnv import StockTradingEnv
from StockEnv.config import INDICATORS as INDICATORS
from GTrXL import GTrXLExtractor

from lstm_policy import CustomActorCriticPolicy, LSTMExtractor, LSTMNetwork

from DRLAgent import DRLAgent
from util import backtest_stats

# 2016-01-04 00:00:00
# 2022-10-14 00:00:00
# 2022-10-17 00:00:00
# 2023-08-16 00:00:00
# 2023-08-17 00:00:00
# 2024-06-28 00:00:00

# DDPG 2.11
# SAC  2.07
# TD3  1.43
# PPO  1.36
# DQN
# A2C  1.87


TRAIN_START_DATE = '2016-01-04'
TRAIN_END_DATE = '2023-8-16'

TRADE_START_DATE = '2023-08-17'
TRADE_END_DATE = '2024-06-28'

stock_nums = os.listdir('/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/股票数据/')
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
dates = df['date'].unique()
tmp_df = pd.DataFrame({'value':range(len(dates))},index=pd.DatetimeIndex(dates))
# 确保每只股票包含相同日期的交易数据
all_dates = pd.date_range(start=latest_start_date, end=earliest_end_date)
df = df.pivot_table(index='date', columns='tic', values=df.columns.difference(['date', 'tic']))
df = df.reindex(tmp_df.index).fillna(method='ffill').fillna(method='bfill')
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
tmp = df.copy()
df = data_split(tmp, TRAIN_START_DATE, TRAIN_END_DATE)
trade_df = data_split(tmp, TRADE_START_DATE, TRADE_END_DATE)

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

PPO_kwargs = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}

SAC_PARAMS = {
    "batch_size": 1,
    "buffer_size": 10000,
    "learning_rate": 0.0001,
    "learning_starts": 10,
    "ent_coef": "auto_0.1",
}

env = StockTradingEnv(df, use_frl_indicators=True,use_predictor=False,use_gtrxl=True,**env_kwargs)
env_train, _ = env.get_sb_env()

env1 = StockTradingEnv(df, use_frl_indicators=True,use_predictor=False,use_gtrxl=False,**env_kwargs)
env1_train, _ = env1.get_sb_env()

trade_env = StockTradingEnv(trade_df, use_frl_indicators=True,use_predictor=False,use_gtrxl=True,**env_kwargs)
env_trade, _ = trade_env.get_sb_env()

policy_kwargs = dict(
    features_extractor_class=GTrXLExtractor,
    features_extractor_kwargs=dict(features_dim=64),  # 根据需要调整
)
GTrXL_PPO = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,  # 将 features_extractor_class 放在 policy_kwargs 中
    verbose=1,
    tensorboard_log="/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/tensorboard/",
    **PPO_kwargs,
)

lstm_policy_kwargs = dict(
    features_extractor_class=LSTMExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

LSTM_PPO = PPO(
    CustomActorCriticPolicy,
    env,
    policy_kwargs=lstm_policy_kwargs,
    verbose=1,
    tensorboard_log="/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/tensorboard/",
    **PPO_kwargs,
    )

agent = DRLAgent(env=env_train)
agent1 = DRLAgent(env=env1_train)

use_sac = 1
use_ppo = 0

if use_ppo:
    # model_ppo = agent.get_model("ppo",model_kwargs = config.PPO_PARAMS, policy_kwargs=policy_kwargs)
    model_ppo = agent.get_model("ppo", model_kwargs=PPO_kwargs, policy=CustomActorCriticPolicy,policy_kwargs=lstm_policy_kwargs)
    # model_ppo = agent1.get_model("ppo", model_kwargs=PPO_kwargs,)
    tmp_path = '/home/czj/pycharm_project_tmp_pytorch/强化学习/rltmp/' + 'GTrXL_PPO' + '/'
    new_logger_gtrxl = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_ppo.set_logger(new_logger_gtrxl)
    trained_ppo = agent.train_model(model=model_ppo, tb_log_name='GTrXL_PPO', total_timesteps=50000)
    trained_moedel = trained_ppo
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_moedel, 
        environment = trade_env)
    print(df_account_value)
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
    perf_stats_all = backtest_stats(df_account_value, value_col_name='account_value')
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_excel('/home/czj/pycharm_project_tmp_pytorch/强化学习/rltmp/' + 'GTrXL_PPO' + '/'+now+'perf_stats_all.xlsx')

if use_sac:
    # model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS, policy_kwargs=policy_kwargs)
    # model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS, policy=CustomActorCriticPolicy,policy_kwargs=lstm_policy_kwargs)
    model_sac = agent1.get_model("sac", model_kwargs=SAC_PARAMS,)
    print(model_sac.policy)
    exit(0)
    tmp_path = '/home/czj/pycharm_project_tmp_pytorch/强化学习/rltmp/' + 'GTrXL_SAC' + '/'
    new_logger_gtrxl = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_sac.set_logger(new_logger_gtrxl)
    trained_sac = agent.train_model(model=model_sac, tb_log_name='GTrXL_SAC', total_timesteps=500000)
    trained_moedl = trained_sac
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_moedl,
        environment = trade_env)
    perf_stats_all = backtest_stats(df_account_value, value_col_name='account_value')
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_excel('/home/czj/pycharm_project_tmp_pytorch/强化学习/rltmp/' + 'GTrXL_SAC' + '/' + now + 'perf_stats_all.xlsx')
