# -*- coding: UTF-8 -*-
from DHC.Poker.Poker_V2.pokerModel_train import *
from utilities.hierarchy.structure.hierarchyReadClass import *

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_set_name = 'chaLearn'
test_folds = [0, 1, 2, 3, 4]
theta = THETA
backbone_name = BACKBONE_NAME
total_folds = TOTAL_FOLDS
epoch_num = EPOCH_NUM
bs = [0.1, 0.01]

data_set_info_dict = get_dataset_info(data_set_name)

hierarchy = HierarchyReadClass(data_set_info_dict['hierarchy_file']).get_hierarchy_info()
label_HLDL_dict = pickle.load(open(data_set_info_dict['HLDL'], 'rb'))

for test_fold in test_folds:
    print('*' * 20, 'Poker begin to deal the fold: ', test_fold, '*' * 20)
    Poker_LDL_train(test_fold, total_folds, data_set_info_dict, hierarchy,
                       label_HLDL_dict, backbone_name, device, epoch_num,
                       'results/chaLearn', 'models/chaLearn', use_wight=True, bs=bs)
    print('*' * 20, 'finish dealt the fold: ', test_fold, '*' * 20, '\n'*3)