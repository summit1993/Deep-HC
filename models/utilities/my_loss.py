# -*- coding: UTF-8 -*-
import torch.nn.functional as F

def My_KL_Loss(predictions, true_distributions):
    predictions = F.log_softmax(predictions, dim=1)
    KL = (true_distributions * predictions).sum()
    KL = -1.0 * KL / predictions.shape[0]
    return KL

def My_KL_Loss_with_Weight(predictions, true_distributions, weight):
    predictions = F.log_softmax(predictions, dim=1)
    KL = (true_distributions * predictions).sum(dim=1)
    KL = -1.0 * KL
    KL = (KL * weight).sum()
    KL = KL / predictions.shape[0]
    return KL