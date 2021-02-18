import torch



x = torch.tensor([[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8]])
print(x)
x = x.flip(1)
print(x)