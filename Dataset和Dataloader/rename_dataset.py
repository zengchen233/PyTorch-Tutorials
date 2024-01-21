import os

root_dir = 'datasets/hymenoptera_data/train'
target_dir = 'bees_image'
img_path = os.listdir(os.path.join(root_dir, target_dir))
label = target_dir.split('_')[0]
# print(label)    # ['ants', 'image']
out_dir = 'bees_label'  # 注意 需要提前建立好此文件夹
for image in img_path:
    filename = image.split('.jpg')[0]
    with open(os.path.join(root_dir, out_dir, '{}.txt'.format(filename)), 'w') as f:
        f.write(label)
