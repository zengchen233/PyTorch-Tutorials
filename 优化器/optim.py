import torch
import torchvision
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
device = torch.device('cuda')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


loss = nn.CrossEntropyLoss()
net = Net()
optim = SGD(net.parameters(), lr=0.01)
for epoch in range(20):
    print('epoch{}'.format(epoch + 1))
    for data in dataloader:
        optim.zero_grad()
        imgs, targets = data
        outputs = net(imgs)
        result = loss(outputs, targets)
        result.backward()
        optim.step()
        print(result)
