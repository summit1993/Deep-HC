# -*- coding: UTF-8 -*-
import pickle
import os

if __name__ == '__main__':
    dir_name = 'results/poker/no_weight'
    files_list = os.listdir(dir_name)
    result = pickle.load(open(os.path.join(dir_name, files_list[0]), 'rb'))
    mae_result = result['MAE']
    for epoch in range(len(mae_result)):
        print(epoch + 1, mae_result[epoch])