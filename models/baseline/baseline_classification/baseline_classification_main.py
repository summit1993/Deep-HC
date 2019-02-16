# -*- coding: UTF-8 -*-
import os
import torch.optim as optim
from torch.utils.data import DataLoader

from baseline.baseline_classification.baseline_classification import *
from utilities.common_tools import MyDataset
from configs.configs import *

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_SET_NAME = 'morph'
DATA_SET_INFO_DICT = get_dataset_info(DATA_SET_NAME)


trainset = MyDataset(DATA_SET_INFO_DICT['info_file'],
                     DATA_SET_INFO_DICT['image_dir'],
                     get_transform_train(DATA_SET_NAME))
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = BaselineClassificationModel(BACKBONE_NAME, DATA_SET_INFO_DICT['label_num'])
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

if __name__ == '__main__':
    running_loss = 0.0
    for epoch in range(EPOCH_NUM):
        for step, data in enumerate(trainloader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % SHOW_ITERS == (SHOW_ITERS - 1):
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1,  running_loss / SHOW_ITERS))
                running_loss = 0.0


