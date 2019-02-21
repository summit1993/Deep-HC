# -*- coding: UTF-8 -*-
import torch
import torch.optim as optim

from baseline.baseline_regression.baseline_regresssion import *
from utilities.data_loader import *
from configs.configs import *
from utilities.my_metrics import *

os.environ["CUDA_VISIBLE_DEVICES"] = "11"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_FOLD = 1
DATA_SET_NAME = 'morph'
DATA_SET_INFO_DICT = get_dataset_info(DATA_SET_NAME)

trainloader, testloader = get_train_test_data_loader(DATA_SET_INFO_DICT,
                                                     TOTAL_FOLDS, TEST_FOLD)

model = BaselineRegressionModel(BACKBONE_NAME)
model = model.to(device)

criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == '__main__':
    log_file = open(DATA_SET_NAME + '_testFold_' + str(TEST_FOLD) + '_results.txt', 'w')
    running_loss = 0.0
    for epoch in range(EPOCH_NUM):
        for step, data in enumerate(trainloader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.transpose(1, 0)[0]
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % SHOW_ITERS == (SHOW_ITERS - 1):
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1,  running_loss / SHOW_ITERS))
                running_loss = 0.0

        print('*' * 10, 'Poker Coming 2333', '*' * 10)
        with torch.no_grad():
            mae_sum = 0.0
            total_count = 0.0
            for _, test_data in enumerate(testloader, 0):
                images, labels = test_data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predicted = outputs.long().transpose(1,0)[0]
                mae_sum += MAE_sum(labels, predicted)
                total_count += len(labels)
            mae = mae_sum / total_count
            print('Epoch: %d, MAE: %.3f' % (epoch + 1, mae))
            log_file.write(str(epoch + 1) + '\t' + str(mae) + '\n')
        print('*' * 10, 'Bye, Poker', '*' * 10, '\n')

    log_file.close()
