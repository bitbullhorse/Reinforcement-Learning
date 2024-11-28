import gym
from ding.torch_utils.network import GTrXL
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.nn import LayerNorm, init, TransformerDecoderLayer, TransformerDecoder, TransformerEncoderLayer


class GTrXLExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64, input_dim:int = 128):
        super(GTrXLExtractor, self).__init__(observation_space, features_dim)
        # 将输入维度映射到GTrXL的输入维度
        self.initial_fc = nn.Linear(observation_space.shape[1], input_dim)
        # 使用GTrXL作为特征提取的一部分
        self.gtrxl = GTrXL(input_dim=input_dim, gru_gating=True, memory_len=12)
        # 额外的全连接层
        # self.final_fc = nn.Linear(observation_space.shape[0], features_dim)
        self.final_fc = nn.Linear(256, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.initial_fc(observations)
        x = self.gtrxl(x, batch_first=True)
        x = self.final_fc(x['logit'])  # 去除第一个维度
        # x = self.final_fc(observations)
        return x[:,-1,:]


class DecoderTransExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64, input_dim:int = 128,  num_layers:int = 6, num_heads:int = 8, d_ff:int = 256, batch_first:bool = True, dtype=torch.float64):
        super(DecoderTransExtractor, self).__init__(observation_space, features_dim)
        self.initial_fc = nn.Linear(observation_space.shape[1], input_dim)
        decoder_layer = TransformerDecoderLayer(input_dim, num_heads, batch_first=batch_first, dtype=dtype, dropout=0.1, dim_feedforward=d_ff)
        decoder_norm = LayerNorm(input_dim, dtype=dtype)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers, norm=decoder_norm)
        self.final_fc = nn.Linear(input_dim, features_dim, dtype=dtype)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.initial_fc(observations)
        x = self.decoder(x, x,tgt_mask=None)[:, -1, :]
        x = self.final_fc(x)
        return x
