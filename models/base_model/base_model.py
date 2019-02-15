from backbone.resnet import *

switch={'resnet-18':lambda :resnet18(pretrained=True),
        'resnet-34': lambda :resnet34(pretrained=True),
        'resnet-50':lambda :resnet50(pretrained=True),
        'resnet-101':lambda :resnet101(pretrained=True),
        'resnet-152': lambda : resnet152(pretrained=True)}

class Base_Model:
    def __init__(self, back_bone_name='resnet-101'):
        self.back_bone = switch[back_bone_name]()
        self.extract_feature_num = self.back_bone.final_feature_num

        # fc = nn.Linear(self.model.final_feature_num, num_classes)
        # self.model.add_module('fc', fc)