import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda')
input = torch.randn(2, 2)
input = torch.reshape(input, (-1, 1, 2, 2))


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, input):
        input = self.sigmoid1(input)
        return input


dataset = torchvision.datasets.CIFAR10('../data', train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, shuffle=True, batch_size=64)

net = MLP()
net.to(device)
step = 0
writer = SummaryWriter('data')
for data in dataloader:
    img, target = data
    writer.add_images('input', img, global_step=step)
    out = net(img)
    writer.add_images('output', out, global_step=step)
    step += 1

writer.close()
