# -*- coding: UTF-8 -*-
import torch.nn.functional as F

def My_KL_Loss(predictions, true_distributions):
    predictions = F.log_softmax(predictions, dim=1)
    KL = (true_distributions * predictions).sum()
    KL = -1.0 * KL / predictions.shape[0]
    return KL