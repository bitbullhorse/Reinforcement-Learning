from ding.torch_utils.network import GTrXL
import torch

# from StockEnv import StockEnv

gtrxl = GTrXL(20, gru_gating=False)

input1= torch.randn(10, 1, 20)

print(gtrxl(input1)['logit'].shape)
print(gtrxl.memory.memory[0][-2])
