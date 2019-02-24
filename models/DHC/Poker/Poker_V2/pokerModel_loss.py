# -*- coding: UTF-8 -*-
from utilities.my_loss import *
import torch
import queue
import numpy as np

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

def pokerModel_calculate_loss_with_weigt_old(outputs, true_labels, hierarchy,
                              label_HLDL_dict, device, bs, is_hard):
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
            if code != -1:
                if is_hard:
                    if true_distribution[-1] >= 0.99:
                        weight[i] = bs[node_depth]
                else:
                    weight[i] = 1.0 - true_distribution[-1] + bs[node_depth]
        true_distributions = true_distributions.to(device)
        weight = weight.to(device)
        total_loss += My_KL_Loss_with_Weight(output, true_distributions, weight)
    return total_loss

def pokerModel_calculate_loss_with_weigt_old_v2(outputs, true_labels, hierarchy,
                              label_HLDL_dict, device, bs, is_hard):
    que = queue.Queue()
    total_loss = 0.0
    samples_count = len(true_labels)
    nodes = hierarchy['nodes']
    score_dict = {}
    score_dict[-1] = torch.zeros(samples_count)
    que.put(-1)
    while not que.empty():
        code = que.get()
        node = nodes[code]
        children_code = node.get_children_code()
        children_count = len(children_code)
        if children_count == 0:
            continue
        code_score = score_dict[code]
        output = outputs[code]
        output_soft_log = F.log_softmax(output.detach().cpu(), dim=1)
        true_distributions = torch.zeros(output.shape)
        if code == -1:
            weight = torch.ones(samples_count)
        else:
            weight = (code_score - min(code_score)) / (max(code_score) - min(code_score))

        for i in range(samples_count):
            true_label = true_labels[i]
            true_distribution = label_HLDL_dict[true_label.item()][code]
            true_distributions[i] = true_distribution

        for j in range(children_count):
            child_score = code_score + output_soft_log[:, j]
            score_dict[children_code[j]] = child_score
            que.put(children_code[j])

        true_distributions = true_distributions.to(device)
        weight = weight.to(device)
        total_loss += My_KL_Loss_with_Weight(output, true_distributions, weight)
    return total_loss


def pokerModel_calculate_loss_with_weigt(outputs, true_labels, hierarchy,
                              label_HLDL_dict, device, bs, is_hard):
    que = queue.Queue()
    total_loss = 0.0
    count = 0.0
    samples_count = len(true_labels)
    nodes = hierarchy['nodes']
    path_dict = hierarchy['paths']
    score_dict = {}
    score_dict[-1] = torch.zeros(samples_count)
    que.put(-1)
    while not que.empty():
        code = que.get()
        node = nodes[code]
        children_code = node.get_children_code()
        children_count = len(children_code)
        if children_count == 0:
            continue
        children_code_set = set(children_code)
        code_score = score_dict[code]
        output = outputs[code]
        output_soft_log = F.log_softmax(output.detach().cpu(), dim=1)
        true_distributions = torch.zeros(output.shape)
        if code == -1:
            weight = torch.ones(samples_count)
        else:
            weight = (code_score - min(code_score)) / (max(code_score) - min(code_score))

        for i in range(samples_count):
            true_label = true_labels[i]
            true_distribution = label_HLDL_dict[true_label.item()][code]
            true_distributions[i] = true_distribution
            if code != -1:
                uset = path_dict[true_label.item()] & children_code_set
                if len(uset) > 0:
                    # weight[i] = 2.0 - weight[i]
                    weight[i] = 1.0

        for j in range(children_count):
            child_score = code_score + output_soft_log[:, j]
            score_dict[children_code[j]] = child_score
            que.put(children_code[j])

        true_distributions = true_distributions.to(device)
        weight = weight.to(device)
        total_loss += My_KL_Loss_with_Weight(output, true_distributions, weight)
        count += 1.0
    total_loss = total_loss / count
    return total_loss
