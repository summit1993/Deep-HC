# -*- coding: UTF-8 -*-
import pickle
import os
import numpy as np
from results.results_analyse.resluts_prediction import *
from utilities.my_metrics import *

def calculate_mae(dir_name):
    mae_sum = 0.0
    file_list = os.listdir(dir_name)
    for file in file_list:
        result = pickle.load(open(os.path.join(dir_name, file), 'rb'))
        mae_sum += np.array(result['MAE'])
    return mae_sum / len(file_list)

def calculate_mae_one_epoch_by_exp(dir_name, file):
    result = pickle.load(open(os.path.join(dir_name, file), 'rb'))
    outputs = result['outputs']
    labels = result['true_labels']
    mae_result = []
    for epoch in range(len(outputs)):
        output_epoch = outputs[epoch]
        labels_epoch = labels[epoch]
        mae_sum = 0.0
        total_count = 0.0
        for i in range(len(output_epoch)):
            output = output_epoch[i]
            true_label = labels_epoch[i]
            prediction = predict_by_Exp(output, 1)
            mae_sum += MAE_sum_np(true_label, prediction)
            total_count += len(true_label)
        mae_mean = mae_sum / total_count
        mae_result.append(mae_mean)
    return mae_result

def calculate_mae_by_exp(dir_name):
    file_list = os.listdir(dir_name)
    mae_sum = 0.0
    for file in file_list:
        mae_sum += np.array(calculate_mae_one_epoch_by_exp(dir_name, file))
    return mae_sum / len(file_list)

if __name__ == '__main__':
    dir_name = 'results/FL'
    mae_mean_exp = calculate_mae_by_exp(dir_name)
    print(min(mae_mean_exp))
    mae_mean_max = calculate_mae(dir_name)
    print(min(mae_mean_max))

    poker_dir_name  = 'results/poker/no_weight'
    poker_mean_max = calculate_mae(poker_dir_name)
    print(min(poker_mean_max))

    regression_dir_name = 'results/FR_V2/L2'
    regression_mean = calculate_mae(regression_dir_name)
    print(min(regression_mean))
