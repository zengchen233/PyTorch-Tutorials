import torch
import torchvision.transforms
from PIL import Image

from CIFAR10 import *

image_path = "../data/imgs/airplane.png"
image = Image.open(image_path)
# print(image)

image = image.convert('RGB')
transforme = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                             torchvision.transforms.ToTensor()])
image = transforme(image)
print(image.shape)

model = CIFAR10()
model_param = torch.load('../模型训练/model/CIFAR10_500.pth', map_location='cuda')
model.load_state_dict(model_param)
# print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output.argmax(1))
