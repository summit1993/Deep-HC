# -*- coding: UTF-8 -*-
from baseline.baseline_regression.baseline_regression_train import *

os.environ["CUDA_VISIBLE_DEVICES"] = "11"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_set_name = 'morph'
test_folds = [2, 3, 4]
theta = THETA
backbone_name = BACKBONE_NAME
total_folds = TOTAL_FOLDS
epoch_num = EPOCH_NUM

dataset_info_dict = get_dataset_info(data_set_name)

for test_fold in test_folds:
    print('*' * 20, 'FR begin to deal the fold: ', test_fold, '*' * 20)
    baseline_Regression(test_fold, total_folds, dataset_info_dict,
                       backbone_name, device, epoch_num, 'results/morph', 'models/morph')
    print('*' * 20, 'finish dealt the fold: ', test_fold, '*' * 20, '\n'*3)