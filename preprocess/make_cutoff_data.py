from math import tan
from os import sysconf
import numpy as np
import scipy.stats as stats
from sklearn import preprocessing
import _pickle as pickle
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
import os

def zscore(array, axis=None, inplace=False):
    """Calculates zscore for an array. A cheap copy of scipy.stats.zscore.
    Inputs:
        array: Numpy array to be normalized
        axis: Axis to operate across [None = entrie array]
        inplace: Do not create new array, change input array [False]
    Output:
        If inplace is True: None
        else: New normalized Numpy-array"""

    if axis is not None and axis >= array.ndim:
        raise np.AxisError('array only has {} axes'.format(array.ndim))

    if inplace and not np.issubdtype(array.dtype, np.floating):
        raise TypeError('Cannot convert a non-float array to zscores')

    mean = array.mean(axis=axis)
    std = array.std(axis=axis)

    if axis is None:
        if std == 0:
            std = 1 # prevent divide by zero

    else:
        std[std == 0.0] = 1 # prevent divide by zero
        shape = tuple(dim if ax != axis else 1 for ax, dim in enumerate(array.shape))
        mean.shape, std.shape = shape, shape

    if inplace:
        array -= mean
        array /= std
        return None
    else:
        return (array - mean) / std

ground_truth = []
with open('../data/truedata/kraken.cleaned.csv', 'r') as f:
    for l in f.readlines():
        items = l.split(',')
        if len(items) == 3:
            continue
        temp = items[0].split('_')
        if int(temp[3]) > 2000:
            ground_truth.append((items[0], int(items[1])))   
truth_dict = {}
for contig, cluster in ground_truth:
    if cluster in truth_dict.keys():
        truth_dict[cluster] += 1
    else:
        truth_dict[cluster] = 0

print('number of types: {}'.format(len(truth_dict)))
tnfs = np.load('../data/truedata/trueout/tnf.npz')
abundances = np.load('../data/truedata/trueout/rpkm.npz')
tnfs = tnfs['arr_0']
abundances = abundances['arr_0']
print(abundances.shape)
zscore(tnfs, axis=0, inplace=True)
zscore(abundances, axis=0, inplace=True)

input_data = np.concatenate([tnfs, abundances], axis=1)

train_set = []
contigs = np.load('../data/truedata/contigs.npz')
contigs = contigs['arr_0']

for idx, val in enumerate(contigs):
    temp = val.split('_')
    if int(temp[3]) >= 2000:
        data = torch.tensor(input_data[idx], dtype=torch.float32)
        # print(data, val, len(contigs), input_data.shape)
        if truth_dict[val] > 10:
            train_set.append((data, val))



print(len(train_set))
print(train_set[:5])
with open("../data/truedata/training_set.pkl", "wb") as f:
    pickle.dump(truth_dict, f)
with open("../data/truedata/truth_dict.pkl", "wb") as f:
    pickle.dump(train_set, f)