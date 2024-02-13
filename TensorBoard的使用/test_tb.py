from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter('logs')
image_path = '../data/hymenoptera_data/train/ants_image/5650366_e22b7e1065.jpg'
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

writer.add_image("test", img_array, 2, dataformats='HWC')

# for i in range(100):
#     writer.add_scalar(tag='y=2x', scalar_value=2*i, global_step=i)  # scalar_value: y轴，global_step：x轴

writer.close()
