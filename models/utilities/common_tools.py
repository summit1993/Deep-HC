# -*- coding: UTF-8 -*-
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
import scipy.io
import pickle
from configs.configs import *

class MyDataset(Dataset):
    def __init__(self, image_list, labels, image_dir, transform, train=True):
        self.transform = transform
        self.image_dir = image_dir
        self.train = train
        self.labels = labels
        self.image_list = image_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image_path = os.path.join(self.image_dir, self.image_list[item])
        img = Image.open(image_path)
        img = self.transform(img)
        if self.train:
            return img, self.labels[item]
        else:
            return img

def get_image_names_and_labels_from_file(file_name):
    image_list = []
    labels = []
    with open(file_name) as f:
        for line in f.readlines():
            line = line.split()
            image_list.append(line[0])
            labels.append(int(line[1]))
    return image_list, labels

def get_train_test_data_loader(data_set_info_dict, total_folds, test_fold):
    image_names_list, labels = get_image_names_and_labels_from_file(data_set_info_dict['info_file'])
    all_index = pickle.load(open(data_set_info_dict['index_file'], 'rb'))
    train_index, test_index = split_data_set(all_index, total_folds, test_fold)
    train_image_names_list = [image_names_list[i] for i in train_index]
    train_labels = [labels[i] for i in train_index]
    test_image_names_list = [image_names_list[i] for i in test_index]
    test_labels = [labels[i] for i in test_index]

    trainset = MyDataset(train_image_names_list, train_labels,
                         data_set_info_dict['image_dir'], get_transform_train(data_set_info_dict['name']))
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testset = MyDataset(test_image_names_list, test_labels,
                        data_set_info_dict['image_dir'], get_transform_test(data_set_info_dict['name']))
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return trainloader, testloader

def split_data_set(all_index, total_folds, test_fold):
    samples_counts = len(all_index)
    fold_size = int(samples_counts / total_folds)
    end_n = min(samples_counts, (test_fold + 1) * fold_size)
    test_index = all_index[test_fold * fold_size: end_n]
    train_index = list(set(all_index) - set(test_index))
    return train_index, test_index

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

def get_dataset_info(imdb_file_name, save_name):
    data = scipy.io.loadmat(imdb_file_name)
    labels = data['images']['label'][0][0][0]
    img_names = data['images']['name'][0][0][0]
    with open(save_name, 'w') as fw:
        for i in range(len(labels)):
            label = str(int(labels[i]))
            img_name = str(img_names[i][0])
            img_name = img_name.split('/')[-1]
            fw.write(img_name + ' ' + label + '\n')
