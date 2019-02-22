# -*- coding: UTF-8 -*-
from utilities.my_loss import *
import torch

def pokerModel_calculate_loss(outputs, true_labels, hierarchy,
                              label_HLDL_dict, device):
    samples_count = len(true_labels)
    total_loss = 0.0
    inners_code_list = hierarchy['inners_code_list'].copy()
    inners_code_list.append(-1)
    for code in inners_code_list:
        output = outputs[code]
        true_distributions = torch.zeros(output.shape)
        for i in range(samples_count):
            true_label = true_labels[i]
            true_distribution = label_HLDL_dict[true_label.item()][code]
            true_distributions[i] = true_distribution
        true_distributions = true_distributions.to(device)
        total_loss += My_KL_Loss(output, true_distributions)
    return total_loss

def pokerModel_calculate_loss_with_weigt(outputs, true_labels, hierarchy,
                              label_HLDL_dict, device, bs):
    samples_count = len(true_labels)
    total_loss = 0.0
    inners_code_list = hierarchy['inners_code_list'].copy()
    inners_code_list.append(-1)
    for code in inners_code_list:
        output = outputs[code]
        true_distributions = torch.zeros(output.shape)
        weight = torch.ones(samples_count)
        node_depth = hierarchy['nodes'][code].get_node_depth()
        for i in range(samples_count):
            true_label = true_labels[i]
            true_distribution = label_HLDL_dict[true_label.item()][code]
            true_distributions[i] = true_distribution
            weight[i] = 1.0 - true_distribution[-1] + bs[node_depth]
        true_distributions = true_distributions.to(device)
        weight = weight.to(device)
        total_loss += My_KL_Loss_with_Weight(output, true_distributions, weight)
    return total_loss