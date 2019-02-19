# -*- coding: UTF-8 -*-
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
from utilities.common_tools import *
import torch

class MyDataset(Dataset):
    def __init__(self, image_list, labels, image_dir, transform, train=True, label_num=-1, has_label_distribution=False, theta=2.0):
        self.transform = transform
        self.image_dir = image_dir
        self.train = train
        self.labels = labels
        self.image_list = image_list
        self.label_num = label_num
        self.theta = theta
        self.has_label_distribution = has_label_distribution

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image_path = os.path.join(self.image_dir, self.image_list[item])
        img = Image.open(image_path)
        img = img.resize((img_size, img_size))
        img = self.transform(img)
        label = self.labels[item]
        if self.train:
            if self.has_label_distribution:
                label_distribution = change_label_2_label_distribution(label, self.label_num, self.theta)
                label_distribution = torch.from_numpy(label_distribution)
                label_distribution = label_distribution.float()
                return img, label_distribution, label
            else:
                return img, label
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
    labels = np.array(labels, dtype=np.int64)
    return image_list, labels

def get_train_test_data_loader(data_set_info_dict, total_folds, test_fold, has_label_distribution=False, theta=2.0, sub_begin_age=True, batch_size=BATCH_SIZE):
    label_num = data_set_info_dict['label_num']
    image_names_list, labels = get_image_names_and_labels_from_file(data_set_info_dict['info_file'])
    if sub_begin_age:
        labels -= data_set_info_dict['begin_age']
    split_index_dict = pickle.load(open(data_set_info_dict['split_index_file'], 'rb'))
    train_image_names_list = []
    train_labels = []
    for fold in range(total_folds):
        if fold != test_fold:
            index_list = split_index_dict[fold]
            tmp_image_list = [image_names_list[i] for i in index_list]
            tmp_label_list = [labels[i] for i in index_list]
            train_image_names_list.extend(tmp_image_list)
            train_labels.extend(tmp_label_list)
    test_image_names_list = [image_names_list[i] for i in split_index_dict[test_fold]]
    test_labels = [labels[i] for i in split_index_dict[test_fold]]

    trainset = MyDataset(train_image_names_list, train_labels,
                         data_set_info_dict['image_dir'], get_transform_train(data_set_info_dict['name']),
                         True, label_num, has_label_distribution, theta)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testset = MyDataset(test_image_names_list, test_labels,
                        data_set_info_dict['image_dir'], get_transform_test(data_set_info_dict['name']),
                        True, label_num, has_label_distribution, theta)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, testloader