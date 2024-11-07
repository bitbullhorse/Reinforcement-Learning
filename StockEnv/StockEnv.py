import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv



class StockTradingEnv(gym.Env):
    def __init__(self,
                 df: pd.DataFrame,
                 stock_dim: int,
                 hmax: int,
                 initial_amount: int,
                 num_stock_shares: list[int],
                 buy_cost_pct: list[float],
                 sell_cost_pct: list[float],
                 reward_scaling: float,
                 state_space: int,
                 action_space: int,
                 tech_indicator_list: list[str],
                 turbulence_threshold=None,
                 risk_indicator_col="turbulence",
                 make_plots: bool = False,
                 print_verbosity=10,
                 day=0,
                 initial=True,
                 previous_state=[],
                 model_name="",
                 mode="",
                 iteration="",
                 ):
        super(StockTradingEnv, self).__init__()
        self.df = df

