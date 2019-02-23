# -*- coding: UTF-8 -*-
from utilities.hierarchy.structure.hierarchyReadClass import *
from utilities.common_tools import change_label_2_label_distribution
import torch
import pickle

def dfs_search(code, code_score_dict, code_LDL_dict, hierarchy, full_label_distribution):
    node = hierarchy['nodes'][code]
    leaf_index_map = hierarchy['leaf_index_map']
    children_code = node.get_children_code()
    children_count = len(children_code)
    if children_count == 0:
        code_score_dict[code] = full_label_distribution[leaf_index_map[code]]
        return
    if code == -1:
        label_distribution = torch.zeros(children_count)
    else:
        label_distribution = torch.zeros(children_count + 1)
    for i in range(children_count):
        child_code = children_code[i]
        dfs_search(child_code, code_score_dict, code_LDL_dict, hierarchy, full_label_distribution)
        label_distribution[i] = code_score_dict[child_code]

    if code == -1:
        code_score_dict[code] = 1.0
        code_LDL_dict[code] = label_distribution
    else:
        sum_item = sum(label_distribution[:-1])
        code_score_dict[code] = sum_item
        label_distribution[-1] = 1.0 - sum_item
        code_LDL_dict[code] = label_distribution

def create_hierarchy_label_distribution(hierarchy_file_name, save_name, theta):
    hierarchy = HierarchyReadClass(hierarchy_file_name).get_hierarchy_info()
    label_HLDL_dict = {}
    leafs_code_list = hierarchy['leafs_code_list']
    label_num = len(leafs_code_list)
    leaf_index_map = hierarchy['leaf_index_map']
    for leaf in leafs_code_list:
        code_score_dict = {}
        code_LDL_dict = {}
        full_label_distribution = change_label_2_label_distribution(leaf_index_map[leaf],
                                                                    label_num, theta)
        dfs_search(-1, code_score_dict, code_LDL_dict, hierarchy, full_label_distribution)
        label_HLDL_dict[leaf] = code_LDL_dict

    pickle.dump(label_HLDL_dict, open(save_name, 'wb'))

if __name__ == '__main__':
    # create_hierarchy_label_distribution('morph_hierarchy.txt', 'morph_HLDL.pkl', 2.0)
    create_hierarchy_label_distribution('chaLearn_hierarchy.txt', 'chaLearn_HLDL_theta.pkl', 2.0)