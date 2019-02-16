# -*- coding: UTF-8 -*-
import torchvision.transforms as transforms
import os

EPOCH_NUM = 50
BATCH_SIZE = 128
BACKBONE_NAME = 'resnet-101'

img_size=224
images_states = {}
images_states['morph'] = {}
images_states['morph']['mean'] = [0.56674693, 0.49170429, 0.46870997]
images_states['morph']['std'] = [0.06605322, 0.05489436, 0.04973615]

def get_dataset_info(data_set_name):
    info_dict = {}
    if data_set_name == 'morph':
        info_dict['label_num'] = 78
        root_dir = 'D:\\program\\deep_learning\\Deep-HC\\Deep-HC\\data\\morph_50000\\'
        info_dict['info_file'] = os.path.join(root_dir, 'morph_50000_info.txt')
        info_dict['image_dir'] = os.path.join(root_dir, 'morph_50000_image')
    return info_dict


def get_transform_train(dataset_name):
    images_state = images_states[dataset_name]
    transform_train = transforms.Compose([
        transforms.RandomCrop(img_size, padding=8),  # 先四周填充0，再把图像随机裁剪成img_size*img_size
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(std=images_state['std'], mean=images_state['mean']),
    ])
    return transform_train

def get_transform_test(dataset_name):
    images_state = images_states[dataset_name]
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(std=images_state['std'], mean=images_state['mean']),
    ])
    return transform_test