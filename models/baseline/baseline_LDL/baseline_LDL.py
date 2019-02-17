# -*- coding: UTF-8 -*-
import torch.nn as nn

from backbone.backbone import *

class BaselineLDLModel(nn.Module):
    def __init__(self, backbone_name, label_num):
        super(BaselineLDLModel, self).__init__()
        self.backbone = Backbone[backbone_name]()
        self.fc = nn.Linear(self.backbone.final_feature_num, label_num)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
