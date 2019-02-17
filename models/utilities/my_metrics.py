# -*- coding: UTF-8 -*-

def MAE_sum(true_labels, predictions):
    error_sum = sum(abs(true_labels - predictions))
    return error_sum.float()

def MAE_mean(true_labels, predictions):
    error_sum = MAE_sum(true_labels, predictions)
    return error_sum / len(true_labels)

