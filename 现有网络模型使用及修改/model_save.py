import torchvision
import torch

vgg_16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1 模型结构以及参数
torch.save(vgg_16, "../data/vgg16_1.pth")

# 保存方式2 模型参数(官方推荐)
torch.save(vgg_16.state_dict(), "../data/vgg16_2.pth")
