# -*- coding: UTF-8 -*-
import torch.nn as nn

from backbone.backbone import *

class PokerModel(nn.Module):
    def __init__(self, backbone_name, hierarchy):
        super(PokerModel, self).__init__()
        self.backbone = Backbone[backbone_name]()
        feature_num = self.backbone.final_feature_num
        self.hierarchy = hierarchy.get_hierarchy_info()
        nodes = self.hierarchy['nodes']
        self.roots_code_list = self.hierarchy['roots_code_list']
        self.inners_code_list = self.hierarchy['inners_code_list']
        self.head_roots = nn.Sequential()
        self.head_inners = nn.Sequential()
        for code in self.roots_code_list:
            node = nodes[code]
            fc = nn.Linear(feature_num, len(node.get_children_code()))
            self.head_roots.add_module(str(code), fc)
        for code in self.inners_code_list:
            node = nodes[code]
            fc = nn.Linear(feature_num, len(node.get_children_code()) + 1)
            self.head_inners.add_module(str(code), fc)

    def forward(self, x):
        x = self.backbone(x)
        outputs = {}
        for code in self.roots_code_list:
            output = self.head_roots.__getattr__(str(code))(x)
            outputs[code] = output
        for code in self.inners_code_list:
            output = self.head_inners.__getattr__(str(code))(x)
            outputs[code] = output
        return outputs
