import torch
import torchvision

model = torch.load("../data/vgg16.pth")
print(model)
