# -*- coding: UTF-8 -*-
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from baseline.baseline_classification.baseline_classification import *
from utilities.common_tools import MyDataset
from configs.configs import *

DATA_SET_NAME = 'morph'
DATA_SET_INFO_DICT = get_dataset_info(DATA_SET_NAME)

trainset = MyDataset(DATA_SET_INFO_DICT['info_file'],
                     DATA_SET_INFO_DICT['image_dir'],
                     get_transform_train(DATA_SET_NAME))
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = BaselineClassificationModel(BACKBONE_NAME, DATA_SET_INFO_DICT['label_num'])

