# -*- coding: UTF-8 -*-
import numpy as np
from torch.utils.data import Dataset
import cv2
import os
import scipy.io

class MyDataset(Dataset):
    def __init__(self, file_name, image_dir, transform, train=True):
        self.transform = transform
        self.image_dir = image_dir
        self.train = train
        self.labels = []
        self.image_list = []
        with open(file_name) as f:
            for line in f.readlines():
                line = line.split()
                self.image_list.append(line[0])
                self.labels.append(int(line[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image_path = os.path.join(self.image_dir, self.image_list[item])
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        if self.train:
            return img, self.labels[item]
        else:
            return img

def calculate_rgb_mean_and_std(img_dir, img_size=224):
    img_list = os.listdir(img_dir)
    img_mean = np.zeros(3)
    img_std = np.zeros(3)
    img_count = len(img_list)
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0
        for i in range(3):
            img_mean[i] = img_mean[i] + img[:, :, i].mean()
    img_mean = img_mean / img_count
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0
        for i in range(3):
            img_std[i] = img_std[i] + ((img[:, :, i] - img_mean[i]) ** 2).sum()
    img_std = img_std / (img_count * img_size * img_size)
    return img_mean, img_std

def get_dataset_info(imdb_file_name, save_name):
    data = scipy.io.loadmat(imdb_file_name)
    labels = data['images']['label'][0][0][0]
    img_names = data['images']['name'][0][0][0]
    with open(save_name, 'w') as fw:
        for i in range(len(labels)):
            label = str(int(labels[i]))
            img_name = str(img_names[i][0])
            fw.write(img_name + ' ' + label + '\n')
