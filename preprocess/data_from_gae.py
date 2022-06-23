from math import tan
from os import sysconf
import numpy as np
import scipy.stats as stats
from sklearn import preprocessing
import _pickle as pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
import os
from cmath import isnan
from torch.nn import functional as F


def cal_similar(data,K, use_gpu=torch.cuda.is_available()):
    
    data = data.cuda()
    N = data.shape[0]
    similar_m = []
    weight_m = []
    for idx in range(N):
        dis = torch.sum(torch.pow(data-data[idx,:],2),dim=1)
        sorted, ind = dis.sort()
        similar_m.append(ind[1:K+1].view(1,K).cpu())
        weight_m.append(Gaussian_kernal(sorted[1:K+1]))
    similar_m = torch.cat(similar_m,dim=0)

    return similar_m, weight_m

def Gaussian_kernal(data, sigma=1):
    # d = data
    if data.shape[0] == 1:
        return 1
    data /= (2*pow(sigma,2))
    data = torch.exp(-data)
    m = nn.Softmax(dim=0)
    # weights = m(data)
    weights =  F.log_softmax(data)
    return weights


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



input_data = np.load('../data/sharon/gcn_embedding.npz')
input_data = input_data['arr_0']
# zscore(input_data, axis=0, inplace=True)
contignames = np.load('../data/sharon/gcn_contignames.npz')
contignames = contignames['arr_0']
# print(contignames.shape, contignames[0])
train_set = []
test_set = []

for data, val in zip(input_data, contignames):
    data = torch.tensor(data)
    test_set.append((data, val))


input_data = torch.from_numpy(input_data)
print(f'init data shape: {input_data.shape}')
similar_m, weight_m = cal_similar(input_data, K=2)
for data, idx_list, weight_list, contigname in zip(input_data, similar_m, weight_m, contignames):
    node = int(contigname.split('_')[1])
    if len(idx_list) == 1:
        train_set.append((data, input_data[idx_list], weight_list, contigname))
    else:
        for idx, weight in zip(idx_list, weight_list):
            train_set.append((data, input_data[idx], weight, contigname))

print(f'After graph embedding: {len(train_set)}')
print(train_set[2][0].shape)
print(train_set[7:10])

print(f"length of test_set{len(test_set)}")

torch.save(train_set, '../data/sharon/gcn_training_set.pkl')
torch.save(test_set, '../data/sharon/gcn_test_set.pkl')