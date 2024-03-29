import torch

x = torch.zeros([3,4])
print(x.shape)
x = x.unsqueeze(0)
print(x.shape)