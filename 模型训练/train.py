import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision
from torch.utils.data import DataLoader
from CIFAR10 import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
train_data_size = len(train_data)
test_data_size = len(test_data)
print('训练数据集长度为:{0},测试数据集长度为:{1}'.format(train_data_size, test_data_size))

# 利用DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
# 网络模型
net = CIFAR10()
net.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss().to(device)
# 优化器
optim = torch.optim.SGD(net.parameters(), lr=1e-2)

# 设置训练网络的一些参数
total_train_step = 0
total_test_step = 0
epoch = 50
# 训练
for i in tqdm(range(epoch)):
    print('------第{}轮训练开始------'.format(i + 1))
    for data in train_dataloader:
        optim.zero_grad()
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = net(imgs)
        loss = loss_fn(output, targets)
        loss.backward()
        optim.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数:{}, Loss:{}".format(total_train_step, loss.item()))

    total_test_loss = 0
    total_acc = 0
    # 测试步骤
    net.eval()
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            test_accuracy = (outputs.argmax(1) == targets).sum()
            total_acc += test_accuracy
    print("整体数据集上的Loss:{}".format(total_test_loss))
    print("整体数据集上的正确率:{}".format(total_acc / test_data_size))
    total_test_step += 1
    torch.save(net.state_dict(), "./model/CIFAR10_{}.pth".format(i))
