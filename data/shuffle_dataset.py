# -*- coding: UTF-8 -*-
import random
import pickle

def shuffle_data_set(samples_num, save_file_name):
    samples_index = list(range(samples_num))
    random.shuffle(samples_index)
    pickle.dump(samples_index, open(save_file_name, 'wb'))

if __name__ == '__main__':
    shuffle_data_set(54736, './morph_50000/samples_index.pkl')

