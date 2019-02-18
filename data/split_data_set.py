# -*- coding: UTF-8 -*-
from collections import defaultdict
import random
import pickle

def analyse_data(data_file_name):
    label_dict = defaultdict(lambda: [])
    with open(data_file_name) as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            line = line.split()
            label_dict[int(line[-1])].append(i)
            i += 1
    return label_dict

def split_data_set(data_file_name, total_fold):
    split_index_dict = {}
    for fold in range(total_fold):
        split_index_dict[fold] = []
    label_dict = analyse_data(data_file_name)
    for label in label_dict:
        label_list = label_dict[label]
        random.shuffle(label_list)
        samples_count = len(label_list)
        count = int(samples_count / total_fold)
        for fold in range(total_fold):
            if count == 0:
                split_index_dict[fold].append(label_list[random.randint(0, samples_count - 1)])
            else:
                begin_fold = fold * count
                if fold < total_fold - 1:
                    end_fold = (fold + 1) * count
                else:
                    end_fold = samples_count
                split_index_dict[fold].extend(label_list[begin_fold:end_fold])
    for fold in split_index_dict:
        random.shuffle(split_index_dict[fold])
    return split_index_dict


if __name__ == '__main__':
    split_index_dict = split_data_set('morph_50000/morph_50000_info.txt', 5)
    pickle.dump(split_index_dict, open('./morph_50000/split_index_dict.pkl', 'wb'))

