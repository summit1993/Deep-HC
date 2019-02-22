# -*- coding: UTF-8 -*-
import torch
import torch.nn.functional as F
import numpy as np

def predict_by_Exp(output, begin_age):
    ages = np.zeros(output.shape)
    for i in range(ages.shape[0]):
        ages[i] = np.arange(begin_age, output.shape[1] + begin_age)
    output = F.softmax(torch.from_numpy(output), dim=1)
    output = output.numpy()
    prediction = ages * output
    prediction = prediction.sum(axis=1)
    prediction = prediction + 0.5
    prediction = prediction.astype('int64')
    return prediction

