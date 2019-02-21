# -*- coding: UTF-8 -*-
import torch.optim as optim

from baseline.baseline_regression.baseline_regresssion import *
from utilities.data_loader import *
from utilities.my_metrics import *
import pickle

def baseline_Regression(test_fold, total_folds, data_set_info_dict,
                       backbone_name, device, epoch_num, results_save_dir, model_save_dir):
    data_set_name = data_set_info_dict['name']
    trainloader, testloader = get_train_test_data_loader(data_set_info_dict,
                                                         total_folds, test_fold)
    model = BaselineRegressionModel(backbone_name)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model_save_dir = os.path.join(model_save_dir, 'testFold_' + str(test_fold))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(results_save_dir):
        os.makedirs(results_save_dir)
    log_file_name = os.path.join(results_save_dir,
                                 'baseline_regression_' + data_set_name + '_testFold_' + str(test_fold) + '_results.pkl')
    baseline_regression_process(model, trainloader, testloader, optimizer, epoch_num, device, log_file_name, model_save_dir)


def baseline_regression_process(model, trainloader, testloader, optimizer, epoch_num, device, log_file_name, model_save_dir):
    results = {}
    results['MAE'] = []
    results['outputs'] = []
    results['predict_labels'] = []
    results['true_labels'] = []
    running_loss = 0.0
    criterion = nn.MSELoss()
    for epoch in range(epoch_num):
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

        if (epoch + 1) % MODEL_SAVE_EPOCH == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(model_save_dir, 'checkpoint_' + str(epoch)))

        print('*' * 10, 'Begin to test', '*' * 10)
        with torch.no_grad():
            mae_sum = 0.0
            total_count = 0.0
            outputs_list = []
            pre_labels_list = []
            true_labels_list = []
            for _, test_data in enumerate(testloader, 0):
                images, labels = test_data
                true_labels_list.append(labels.numpy())
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs_list.append(outputs.cpu().numpy())
                predicted = outputs.long().transpose(1,0)[0]
                pre_labels_list.append(predicted.cpu().numpy())
                mae_sum += MAE_sum(labels, predicted)
                total_count += len(labels)
            mae = mae_sum / total_count
            mae = mae.item()
            results['MAE'].append(mae)
            results['outputs'].append(outputs_list)
            results['predict_labels'].append(pre_labels_list)
            results['true_labels'].append(true_labels_list)
            print('Epoch: %d, MAE: %.3f' % (epoch + 1, mae))
        print('*' * 10, 'Finish test', '*' * 10, '\n')

    pickle.dump(results, open(log_file_name, 'wb'))
