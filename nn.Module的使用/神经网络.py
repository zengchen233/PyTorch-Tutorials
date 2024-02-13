import torch.nn as nn
import torch


class zcc(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x += 1
        return x


zc = zcc()
x = torch.tensor(0)
output = zc(x)
print(output)
