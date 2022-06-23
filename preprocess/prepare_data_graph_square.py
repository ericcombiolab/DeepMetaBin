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
from collections import defaultdict
import scipy.sparse as sp


def cal_similar(data,K, use_gpu=torch.cuda.is_available()):
    if use_gpu:
        data = data.cuda()
    N = data.shape[0]
    similar_m = []
    weight_m = []
    for idx in range(N):
        diff = torch.pow(data-data[idx,:],2)
        diff[:,-1] *= 50
        # print(diff.shape)
        dis = torch.sum(diff,dim=1)
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
    weights = m(data)
    # weights =  F.log_softmax(data)
    return weights

def dict2csr(X_dict):
    keys = X_dict.keys()
    map_dict = dict(zip(list(keys), range(len(keys))))
    rows, cols, vals = [], [], []
    for key, values in X_dict.items():
        for value in values:
            rows.append(map_dict[key])
            cols.append(map_dict[value[0]])
            vals.append(value[1])
    X = sp.csr_matrix((vals, (rows, cols)))
    return X

def csr2dict(X_csr, keys):
    map_dict_reverse = dict(zip(range(len(keys)), list(keys)))

    Xcoo = X_csr.tocoo() # convert csr matrix to coo sparse matrix
    x_convert = defaultdict(list)
    for (r, c, d) in zip(Xcoo.row, Xcoo.col, Xcoo.data):
        x_convert[map_dict_reverse[r]].append((map_dict_reverse[c] , d))
    x_convert = dict(x_convert)
    x_dict = dict(zip(list(keys), range(len(keys))))
    for key in x_dict.keys():
        if key in x_convert.keys():
            x_dict[key] = x_convert[key]
        else:
            x_dict[key] = []
    return x_dict

def makeXdict(idx_list, weight_list):
    X_dict = defaultdict(list)
    idx = 0
    for key_list, value_list in zip(idx_list,weight_list):
        if len(key_list)==0:
            X_dict[idx] = [(idx, 0.0)]
        for key,weight in zip(key_list, value_list):
            X_dict[idx].append((key, weight))
        idx += 1
    return dict(X_dict)

def csr_matrix_mul(X_csr, K = 2):
    X_csr = X_csr.toarray()
    print(X_csr.shape)
    for i in range(K-1):
        X_csr = np.matmul(X_csr, X_csr)
    for i in range(X_csr.shape[0]):
        X_csr[i,i] = 0.0
    X_csr = np.stack(X_csr[i]/np.sum(X_csr[i]) for i in range(X_csr.shape[0]))
    X_csr[np.isnan(X_csr)] = 0.0
    return sp.csr_matrix(X_csr)

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

def plot_latent_space(features, labels, save=False):
    
    # plot only the first 2 dimensions
    fig = plt.figure(figsize=(8, 6))
    types = set()
    for i in labels:
        types.add(i)
    print(len(types))
    plt.scatter(features[:, 0], features[:, 1], c=labels, marker='o',
            edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s = 10)
    plt.colorbar()
    if(save):
        fig.savefig('vamb_latent_space.png')
    return fig


tnfs = np.load('../../DeepBin/data/CAMI1_L/vamb/tnf.npz')
abundances = np.load('../../DeepBin/data/CAMI1_L/vamb/rpkm.npz')
tnfs = tnfs['arr_0']
abundances = abundances['arr_0']

#------------------------------
zscore(tnfs, axis=0, inplace=True)
zscore(abundances, axis=0, inplace=True)
# -----------------------------
input_data = np.concatenate([tnfs, abundances], axis=1)

# -----------------------------

train_set = []
test_set = []
contigs = np.load('../../DeepBin/data/CAMI1_L/vamb/contignames.npz')
contigs = contigs['arr_0']
graph = torch.load('../data/CAMI1_L/ag_dict.pkl')
print(graph[1000])
init_data = []
contignames = []
node_num = {}
# node_num = []
for idx, val in enumerate(contigs):
    temp = val.split('_')
    if int(temp[3]) >= 1000:
        data = torch.tensor(input_data[idx], dtype=torch.float32)
        init_data.append(data)
        node_num[int(temp[1])] = data
        # node_num.append(int(temp[1]))
        contignames.append(val)
        test_set.append((data, val))

init_data = torch.stack(init_data, dim=0)
print(f'init data shape: {init_data.shape}')
similar_m0, weight_m0 = cal_similar(init_data, K=15)
similar_m = []
weight_m = []
idx = 0

for data, idx_list, weight_list, contigname in zip(init_data, similar_m0, weight_m0, contignames):
    node_idx = int(contigname.split('_')[1])
    new_idx = []
    new_weight = []
    if node_idx in graph.keys():
        graph_nodes = graph[node_idx]
    else:
        similar_m.append(new_idx)
        weight_m.append(new_weight)
        continue
    for i, weight in zip(idx_list, weight_list):
        # print(i)
        for num in graph_nodes:
            # print(1)
            if (init_data[i] == node_num[num]).sum() == 104:
                # print(1)
                new_idx.append(i.item())
                new_weight.append(weight.item())
                break
    similar_m.append(new_idx)
    weight_m.append(new_weight)       
    idx += 1

X_dict = makeXdict(similar_m, weight_m)
X_csr = dict2csr(X_dict)
X_csr = csr_matrix_mul(X_csr)
X_dict = csr2dict(X_csr,X_dict.keys())

for data, edges, contigname in zip(init_data, X_dict.values(), contignames):
    node = int(contigname.split('_')[1])
    if len(edges) == 0:
        train_set.append((data, data, torch.tensor(0.5), contigname))
    else:
        # print(1)
        for idx, weight in edges:
            # print(idx)
            train_set.append((data, init_data[idx], torch.tensor(weight), contigname))
    # if node in graph.keys():
    #     for num in graph[node]:
    #         w = torch.tensor(1)
    #         if torch.cuda.is_available():
    #             w = w.cuda()
    #         train_set.append((data, node_num[num], w, contigname))


# ---------------------------------

print(f'After graph embedding: {len(train_set)}')
# print(train_set[2][0].shape)
# print(train_set[7:10])

print(f"length of test_set{len(test_set)}")

torch.save(train_set, '../data/CAMI1_L/training_set.pkl')
torch.save(test_set, '../data/CAMI1_L/test_set.pkl')

# with open("../data/sharon/test_set1.pkl", "wb") as f:
#     pickle.dump(test_set, f)