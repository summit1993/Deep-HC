import numpy as np
import torch

def change_label_2_label_distribution(label, label_num, theta):
    A = np.arange(label_num)
    A -= label
    item1 = 1.0 / ((2.0 * np.pi) ** 0.5 * theta)
    item2 = A ** 2 / (2.0 * (theta ** 2))
    item2 = -1.0 * item2
    label_distribution = item1 * np.exp(item2)
    return label_distribution

if __name__ == '__main__':
    label = 3
    label_num = 20
    theta = 2.0
    label_distribution = change_label_2_label_distribution(label, label_num, theta)
    print(label_distribution)
    label_distribution_tensor = torch.from_numpy(label_distribution)
    print(label_distribution_tensor.dtype)
    label_distribution_tensor = label_distribution_tensor.float()
    print(label_distribution_tensor.dtype)
