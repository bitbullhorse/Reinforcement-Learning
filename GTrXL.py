from ding.torch_utils.network import GTrXL
import torch

gtrxl = GTrXL(20)

input1= torch.randn(12, 10, 20)

print(gtrxl(input1)['logit'][-1,:].shape)


