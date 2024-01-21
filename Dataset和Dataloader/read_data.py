from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)  # 将地址进行拼接
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


ants_dataset = MyData(root_dir='datasets/hymenoptera_data/train', label_dir='ants')
img, label = ants_dataset[2]
print(img, label)   # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x282 at 0x1F774B1C580> ants

bee_dataset = MyData(root_dir='datasets/hymenoptera_data/train', label_dir='bees')
img, label = bee_dataset[2]
print(img, label)   # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=464x500 at 0x1F72BFA1CD0> bees

train_data = ants_dataset + bee_dataset  # 合并数据集
print(train_data)   # <torch.utils.data.dataset.ConcatDataset object at 0x000001F774B1C580>
