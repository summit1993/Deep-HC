# -*- coding: UTF-8 -*-
from utilities.my_loss import *
import torch

def pokerModel_calculate_loss(outputs, true_labels, hierarchy, gamma, device):
    inners_code_list = hierarchy['inners_code_list'].copy()
    nodes = hierarchy['nodes']
    path_dict = hierarchy['paths']
    inners_code_list.append(-1)
    samples_count = len(true_labels)
    total_loss = 0.0
    for code in inners_code_list:
        node = nodes[code]
        output = outputs[code]
        children_code = node.get_children_code()
        children_code_set = set(children_code)
        weight = torch.ones(samples_count)
        true_distributions = torch.zeros(output.shape)
        for i in range(samples_count):
            true_label = true_labels[i]
            uset = path_dict[true_label.item()] & children_code_set
            if len(uset) == 0:
                weight[i] = (1.0 - output[i][-1]) ** gamma
                true_distributions[i][-1] = 1.0
            else:
                ulabel = list(uset)[0]
                uindex = children_code.index(ulabel)
                true_distributions[i][uindex] = 1.0
        true_distributions = true_distributions.to(device)
        weight = weight.to(device)
        total_loss += My_KL_Loss_with_Weight(output, true_distributions, weight)
    return total_loss
