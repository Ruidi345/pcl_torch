import torch

# x = torch.zeros(2,2)
x1 = torch.arange(start=0, end=6, step=1)
print(x1.shape)
# x = x.reshape(2,3)

# print(x[0].size())
# x[1] = torch.arange(start=100, end=103, step=1)
x = torch.empty(2,2)
expand = torch.unsqueeze(x, 0)
expand = torch.cat((expand, torch.ones(1,2,2)))
expand = torch.cat((expand, torch.ones(1,2,2)))
print(expand.size())
print(expand)
print(torch.sum(expand, dim=0))