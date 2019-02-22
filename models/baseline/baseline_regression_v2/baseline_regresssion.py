# -*- coding: UTF-8 -*-
import torch.nn as nn

from backbone.backbone import *

class BaselineRegressionModel(nn.Module):
    def __init__(self, backbone_name):
        super(BaselineRegressionModel, self).__init__()
        self.backbone = Backbone[backbone_name]()
        self.fc = nn.Linear(self.backbone.final_feature_num, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = self.tanh(x)
        return x