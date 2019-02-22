# -*- coding: UTF-8 -*-
import numpy as np
from PIL import Image
import scipy.io
from configs.configs import *

def split_data_set(all_index, total_folds, test_fold):
    samples_counts = len(all_index)
    fold_size = int(samples_counts / total_folds)
    if test_fold < total_folds - 1:
        end_n = (test_fold + 1) * fold_size
    else:
        end_n = samples_counts
    test_index = all_index[test_fold * fold_size: end_n]
    train_index = list(set(all_index) - set(test_index))
    return train_index, test_index

def change_label_2_label_distribution(label, label_num, theta):
    if theta == 0:
        A = np.zeros(label_num)
        A[label] = 1.0
        return A
    A = np.arange(label_num)
    A -= label
    item1 = 1.0 / ((2.0 * np.pi) ** 0.5 * theta)
    item2 = A ** 2 / (2.0 * (theta ** 2))
    item2 = -1.0 * item2
    label_distribution = item1 * np.exp(item2)
    return label_distribution

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
        img = img.resize((img_size, img_size))
        img = np.array(img, dtype=np.float32)
        img = img / 255.0
        for i in range(3):
            img_mean[i] = img_mean[i] + img[:, :, i].mean()
    img_mean = img_mean / img_count
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        img = img.resize((img_size, img_size))
        img = np.array(img, dtype=np.float32)
        img = img / 255.0
        for i in range(3):
            img_std[i] = img_std[i] + ((img[:, :, i] - img_mean[i]) ** 2).sum()
    img_std = img_std / (img_count * img_size * img_size)
    return img_mean, img_std

def get_dataset_info_from_mat(imdb_file_name, save_name):
    data = scipy.io.loadmat(imdb_file_name)
    labels = data['images']['label'][0][0][0]
    img_names = data['images']['name'][0][0][0]
    with open(save_name, 'w') as fw:
        for i in range(len(labels)):
            label = str(int(labels[i]))
            img_name = str(img_names[i][0])
            img_name = img_name.split('/')[-1]
            fw.write(img_name + ' ' + label + '\n')
