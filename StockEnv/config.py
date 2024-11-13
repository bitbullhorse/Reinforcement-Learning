import json

INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
    '复权价1(元)_AdjClpr1', 
    '复权价2(元)_AdjClpr2', 
    '市盈率_PE',
    '总市值加权平均市场日收益率_Dretmc', 
    '总市值加权平均日资本收益_Daretmc', 
    '总股数日换手率(%)_DFulTurnR',
    '成交金额_Trdsum', 
    '日振幅(%)_Dampltd', 
    '日收益率_Dret', 
    '日无风险收益率_DRfRet',
    '日资本收益率_Daret', 
    '流通市值加权平均市场日收益率_Drettmv', 
    '流通股日换手率(%)_DTrdTurnR',
    '等权平均市场日收益率_Dreteq', 
    '等权平均市场日资本收益率_Dareteq',
]

INDEX = ['日期_Date', 
         '收盘价_Clpr', 
         '开盘价_Oppr', 
         '最高价_Hipr', 
         '最低价_Lopr', 
         '复权价1(元)_AdjClpr1', 
         '复权价2(元)_AdjClpr2',
         '成交量_Trdvol',
         '成交金额_Trdsum',
         '日振幅(%)_Dampltd',
         '总股数日换手率(%)_DFulTurnR', 
         '流通股日换手率(%)_DTrdTurnR', 
         '日收益率_Dret', 
         '日资本收益率_Daret',
         '等权平均市场日收益率_Dreteq',
         '流通市值加权平均市场日收益率_Drettmv', 
         '总市值加权平均市场日收益率_Dretmc', 
         '等权平均市场日资本收益率_Dareteq',
         '总市值加权平均日资本收益_Daretmc', 
         '日无风险收益率_DRfRet', 
         '市盈率_PE']

STOCK_NUMS =[
              '601628/',
              '600048/',
              '600030/',
              '600015/',
              '601318/',
              '600000/',
              '601088/',
              '600031/',
              '601688/',
              '601668/',
              '601328/',
              '603288/',
              '600519/',
              '600150/',
              '601818/',
              '600018/',
              '600028/',
              '600585/',
              '600372/',
              '601398/',
              '601288/',
              '601901/',
              '600050/',
              '601919/',
              '600010/',
              '601857/',
              '000001/',
              '600016/',
              '600809/',
              '601601/',
              '600036/',
              '600104/', 
            ]

DATA_PATH = '/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/股票数据/'

TRAINED_STOCKS_FILE = '/home/czj/pycharm_project_tmp_pytorch/强化学习/StockEnv/trained_stocks.json'

# 加载已训练的股票列表
try:
    with open(TRAINED_STOCKS_FILE, 'r') as f:
        TRAINED_STOCKS = json.load(f)
except FileNotFoundError:
    TRAINED_STOCKS = []

def save_trained_stocks():
    with open(TRAINED_STOCKS_FILE, 'w') as f:
        json.dump(TRAINED_STOCKS, f)

def add_trained_stock(stock_num):
    if stock_num not in TRAINED_STOCKS:
        TRAINED_STOCKS.append(stock_num)
        save_trained_stocks()

def is_stock_trained(stock_num):
    return stock_num in TRAINED_STOCKS
