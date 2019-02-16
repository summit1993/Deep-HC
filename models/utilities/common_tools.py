# -*- coding: UTF-8 -*-
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import os

def calculate_rgb_mean_and_std(img_dir):
    img_list = os.listdir(img_dir)
    img_size = 224
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