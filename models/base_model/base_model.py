from backbone.resnet import *

switch={'resnet-18':lambda num_classes:resnet18(pretraine=True, num_classes=num_classes),
        'resnet-34': lambda num_classes:resnet34(pretraine=True, num_classes=num_classes),
        'resnet-50':lambda num_classes:resnet50(pretraine=True, num_classes=num_classes),
        'resnet-101':lambda num_classes:resnet101(pretraine=True, num_classes=num_classes),
        'resnet-152': lambda num_classes: resnet152(pretraine=True, num_classes=num_classes)}

class Base_Model:
    def __init__(self, back_bone_name='resnet-101', num_classes=-1):
        self.back_bone = switch[back_bone_name](num_classes)
        # here to go