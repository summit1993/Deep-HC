# -*- coding: UTF-8 -*-
import torch
import pickle
import torch.optim as optim
from torch.utils.data import DataLoader

from baseline.baseline_classification.baseline_classification import *
from utilities.common_tools import *
from configs.configs import *
from utilities.my_metrics import *

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_FOLD = 0

DATA_SET_NAME = 'morph'
DATA_SET_INFO_DICT = get_dataset_info(DATA_SET_NAME)


image_names_list, labels = get_image_names_and_labels_from_file(DATA_SET_INFO_DICT['info_file'])
all_index = pickle.load(open(DATA_SET_INFO_DICT['index_file'], 'rb'))
train_index, test_index = split_data_set(all_index, TOTAL_FOLDS, TEST_FOLD)
train_image_names_list = [image_names_list[i] for i in train_index]
train_labels = [labels[i] for i in train_index]
test_image_names_list = [image_names_list[i] for i in test_index]
test_labels = [labels[i] for i in test_index]

trainset = MyDataset(train_image_names_list, train_labels,
                     DATA_SET_INFO_DICT['image_dir'], get_transform_train(DATA_SET_NAME))
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
testset = MyDataset(test_image_names_list, test_labels,
                    DATA_SET_INFO_DICT['image_dir'], get_transform_test(DATA_SET_NAME))
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = BaselineClassificationModel(BACKBONE_NAME, DATA_SET_INFO_DICT['label_num'])
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)

log_file = open(DATA_SET_NAME + '_testFold_' + str(TEST_FOLD) +'_results.txt', 'w')

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

        print('*'*10, 'Poker Coming 2333', '*'*10)
        with torch.no_grad():
            mae_sum = 0.0
            for _, test_data in enumerate(testloader, 0):
                images, labels = test_data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                mae_sum += MAE_sum(labels, predicted)
            mae = mae_sum / len(test_labels)
            print('Epoch: %d, MAE: %.3f'%(epoch + 1, mae))
            log_file.write(str(epoch + 1) + '\t' + str(mae) + '\n')
        print('*' * 10, 'Bye, Poker', '*' * 10, '\n')

    log_file.close()
