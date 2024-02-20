import torch
import torchvision.datasets
from torch import nn

train_data = torchvision.datasets.CIFAR10('../data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
# 预训练模型
vgg_16 = torchvision.models.vgg16(pretrained=False)
vgg_16.classifier[6] = nn.Linear(4096, 10)
torch.save(vgg_16, "../data/vgg16.pth")
print(vgg_16)
