import scipy.io
from utilities.common_tools import *

# item = 3
#
# data = scipy.io.loadmat('morph_imdb.mat')
#
# labels = data['images']['label'][0][0][0]

# label = int(labels[item])
#
# names = data['images']['name'][0][0][0]
#
# name = str(names[item][0])
# print(label, name)

get_dataset_info('morph_imdb.mat', 'morph_info.txt')

with open('morph_info.txt') as f:
    lines = f.readlines()
    print(lines[0])