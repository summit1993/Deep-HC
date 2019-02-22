# -*- coding: UTF-8 -*-
import pickle
import os
import numpy as np

# dir_name = 'results/poker/no_weight'
dir_name = 'results/FL'
file_list = os.listdir(dir_name)

mae_mean = 0.0

for file in file_list:
    result = pickle.load(open(os.path.join(dir_name, file), 'rb'))
    mae_mean += np.array(result['MAE'])

mae_mean /= len(file_list)

print(min(mae_mean))
