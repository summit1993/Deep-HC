# -*- coding: UTF-8 -*-
import torch.optim as optim

from DHC.Poker.pokerModel import *
from DHC.Poker.pokerModel_loss import *
from DHC.Poker.pokerModel_prediction import *
from utilities.data_loader import *
from configs.configs import *
from utilities.my_metrics import *
from utilities.hierarchy.structure.hierarchyReadClass import *

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 256
backbone_name = 'resnet-50'
gamma = 2.0
poker_thetas = POKER_THETAS
TEST_FOLD = 0
DATA_SET_NAME = 'morph'
DATA_SET_INFO_DICT = get_dataset_info(DATA_SET_NAME)

trainloader, testloader = get_train_test_data_loader(DATA_SET_INFO_DICT,
                                                     TOTAL_FOLDS, TEST_FOLD,
                                                     sub_begin_age=False,
                                                     batch_size=batch_size)

hierarchy = HierarchyReadClass(DATA_SET_INFO_DICT['hierarchy_file']).get_hierarchy_info()

model = PokerModel(backbone_name, hierarchy)
model = model.to(device)

# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == '__main__':
    log_file = open(DATA_SET_NAME + '_testFold_' + str(TEST_FOLD) + '_resnet34_results.txt', 'w')
    running_loss = 0.0
    for epoch in range(EPOCH_NUM):
        for step, data in enumerate(trainloader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = pokerModel_calculate_loss(outputs, labels, hierarchy, gamma, device,
                                             True, poker_thetas)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % SHOW_ITERS == (SHOW_ITERS - 1):
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, step + 1, running_loss / SHOW_ITERS))
                running_loss = 0.0

        print('*' * 10, 'Poker Coming 2333', '*' * 10)
        with torch.no_grad():
            mae_sum = 0.0
            total_count = 0.0
            for _, test_data in enumerate(testloader, 0):
                images, labels = test_data
                labels -= DATA_SET_INFO_DICT['begin_age']
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = pokerModel_prediction(outputs, hierarchy)
                outputs = outputs.to(device)
                _, predicted = torch.max(outputs.data, 1)
                mae_sum += MAE_sum(labels, predicted)
                total_count += len(labels)
            mae = mae_sum / total_count
            print('Epoch: %d, MAE: %.3f' % (epoch + 1, mae))
            log_file.write(str(epoch + 1) + '\t' + str(mae) + '\n')
        print('*' * 10, 'Bye, Poker', '*' * 10, '\n')

    log_file.close()
