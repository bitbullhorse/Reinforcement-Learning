from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import torch

class LSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64, input_dim:int = 128):
        super(LSTMExtractor, self).__init__(observation_space, features_dim)
        # 将输入维度映射到LSTM的输入维度
        self.initial_fc = nn.Linear(observation_space.shape[1], input_dim)
        # 使用LSTM作为特征提取的一部分
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=128, num_layers=3, batch_first=True)
        # 额外的全连接层
        # self.final_fc = nn.Linear(observation_space.shape[0], features_dim)
        self.final_fc = nn.Linear(128, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.initial_fc(observations)
        x, _ = self.lstm(x)
        x = self.final_fc(x)  # 去除第一个维度
        # x = self.final_fc(observations)
        return x


class LSTMNetwork(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
            layer_num: int = 3,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.LSTM(feature_dim, self.latent_dim_pi, layer_num, batch_first=True)
        
        # Value network
        self.value_net = nn.LSTM(feature_dim, self.latent_dim_vf, layer_num, batch_first=True)
    
    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)
    
    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)[0][:,-1,:]
    
    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)[0][:,-1,:]
    

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            *args,
            **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )    
    
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = LSTMNetwork(self.features_dim)
