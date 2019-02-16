# -*- coding: UTF-8 -*-
from backbone.resnet import *

Backbone = {'resnet-18':lambda :resnet18(pretrained=True),
        'resnet-34': lambda :resnet34(pretrained=True),
        'resnet-50':lambda :resnet50(pretrained=True),
        'resnet-101':lambda :resnet101(pretrained=True),
        'resnet-152': lambda : resnet152(pretrained=True)}

# if __name__ == '__main__':
#     back_bone = Backbone['resnet-101']()
#     extract_feature_num =back_bone.final_feature_num
    # fc = nn.Linear(backbone.final_feature_num, num_classes)
    # self.model.add_module('fc', fc)