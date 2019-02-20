# -*- coding: UTF-8 -*-
import queue
import torch
import torch.nn.functional as F

def pokerModel_prediction(outputs, hierarchy):
    que = queue.Queue()
    leaf_index_map = hierarchy['leaf_index_map']
    roots_code_list = hierarchy['roots_code_list']
    path_dict = hierarchy['paths']
    nodes = hierarchy['nodes']
    code_score_dict = {}
    root_output = F.softmax(outputs[-1], dim=1)
    root_output = torch.log(root_output)
    predictions = torch.zeros(root_output.shape[0], len(leaf_index_map))
    for i in range(len(roots_code_list)):
        code_score_dict[roots_code_list[i]] = root_output[:, i]
    for code in roots_code_list:
        que.put(code)
    while not que.empty():
        code = que.get()
        node = nodes[code]
        children_code_list = node.get_children_code()
        if len(children_code_list) > 0:
            output = F.softmax(outputs[code], dim=1)
            output = output[:, :-1]
            output = torch.log(output)
            children_scores = output.transpose(1, 0) + code_score_dict[code]
            children_scores = children_scores.transpose(1, 0)
            for i in range(len(children_code_list)):
                child = children_code_list[i]
                code_score_dict[child] = children_scores[:, i]
                que.put(child)
        else:
            predictions[:, leaf_index_map[code]] = code_score_dict[code] / len(path_dict[code])
    return predictions
