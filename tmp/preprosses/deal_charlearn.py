import os
import numpy as np
from PIL import Image

def calculate_rgb_mean_and_std(img_dir, img_size=224):
    img_list = os.listdir(img_dir)
    img_mean = np.zeros(3)
    img_std = np.zeros(3)
    img_count = len(img_list)
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (img_size, img_size))
        img = Image.open(img_path)
        img = np.array(img, dtype=np.float32)
        img = img / 255.0
        for i in range(3):
            img_mean[i] = img_mean[i] + img[:, :, i].mean()
    img_mean = img_mean / img_count
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        img = np.array(img, dtype=np.float32)
        img = img / 255.0
        for i in range(3):
            img_std[i] = img_std[i] + ((img[:, :, i] - img_mean[i]) ** 2).sum()
    img_std = img_std / (img_count * img_size * img_size)
    return img_mean, img_std

def move_image(file_name, img_dir, save_img_dir, img_size=224):
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            img = Image.open(os.path.join(img_dir, line[0]))
            img = img.resize((img_size, img_size))
            img.save(os.path.join(save_img_dir, line[0]), quality=95, subsampling=0)

if __name__ == '__main__':
    # # move_image('./charLearn/16_valid_gt.txt', './charLearn/16_val', './charLearn/16_val_resize')
    # results = calculate_rgb_mean_and_std('charLearn/16_all_resize')
    # print(results)

    # move_image('morph_50000_info.txt', 'D:\\program\\deep_learning\\Deep-HC\\Deep-HC\\data\\morph_50000\\morph_50000_image',
    #            'morph_50000_image_resized')
    results = calculate_rgb_mean_and_std('morph_50000_image_resized')
    print(results)