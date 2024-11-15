import gym
from ding.torch_utils.network import GTrXL
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
class GTrXLExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64):
        super(GTrXLExtractor, self).__init__(observation_space, features_dim)
        # 将输入维度映射到GTrXL的输入维度
        self.initial_fc = nn.Linear(observation_space.shape[0], 20)
        # 使用GTrXL作为特征提取的一部分
        self.gtrxl = GTrXL(input_dim=20, gru_gating=False)
        # 额外的全连接层
        # self.final_fc = nn.Linear(observation_space.shape[0], features_dim)
        self.final_fc = nn.Linear(256, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.initial_fc(observations)
        x = self.gtrxl(x.unsqueeze(0), batch_first=True)
        x = self.final_fc(x['logit'].squeeze(0))  # 去除第一个维度
        # x = self.final_fc(observations)
        return x
