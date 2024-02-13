import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10('../data', train=False, transform=torchvision.transforms.ToTensor())

# 一次性拿进去64张图片进行训练
loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, drop_last=True)

writer = SummaryWriter('dataloader')
step = 0

for data in loader:
    img, target = data
    writer.add_images("test_data_drop_last", img, step)
    step += 1

writer.close()
