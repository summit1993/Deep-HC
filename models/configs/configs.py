# -*- coding: UTF-8 -*-
import torchvision.transforms as transforms

img_size=224
images_states = {}
images_states['morph'] = {}
images_states['morph']['mean'] = [0.56674693, 0.49170429, 0.46870997]
images_states['morph']['std'] = [0.06605322, 0.05489436, 0.04973615]

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