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


def load_large_obj(input_file_path):
    bytes_in = bytearray(0)
    max_bytes = 2 ** 31 - 1
    input_size = os.path.getsize(input_file_path)
    with open(input_file_path, 'rb') as f_in:
        for i in range(0, input_size, max_bytes):
            size = min(max_bytes, input_size - i)
            bytes_in += f_in.read(size)
    obj = pickle.loads(bytes_in)
    print("Finish loading the dict")
    return obj

def store_large_obj(obj, output_file_path):
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(obj)
    with open(output_file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            size = min(max_bytes, len(bytes_out) - idx)
            f_out.write(bytes_out[idx:idx + size])
    print("Finish storing the dict")

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

# ------------to plot-------------
# latents = np.load('../data/outdata/latent.npz')
# latents = latents['arr_0']
ground_truth = []
with open('../data/sharon/kraken.cleaned.csv', 'r') as f:
    # cnt = 0
    for l in f.readlines():
        items = l.split(',')
        if len(items) == 3:
            continue
        temp = items[0].split('_')
        if int(temp[3]) >= 1000:
            # print(items[1])
            # if int(items[1]) == 853:
            #     cnt += 1
            ground_truth.append((items[0], items[1]))
    # print(cnt)
# contigs = []
# with open("../data/contigs.pkl", "rb") as f:
#     contig_list = pickle.load(f)
#     for val in contig_list:
#         temp = val.split('_')
#         if int(temp[3]) >= 100:
#             contigs.append(val)
label_dic = {}
label_types = set()
types_dict = {}
for contigname, category in ground_truth:
    label_types.add(category)
    label_dic[contigname] = category
print('types: {}'.format(len(label_types)))
# for idx, val in enumerate(label_types):
#     types_dict[val] = idx
# for key, value in label_dic.items():
#     label_dic[key] = types_dict[value]
# true_labels = []
# del_idx = []
# for idx, contig in enumerate(contigs):
#     try:
#         true_labels.append(label_dic[contig])
#     except KeyError:
#         del_idx.append(idx)
# true_labels = np.array(true_labels)
# temp = np.zeros(latents.shape[0]) 
# temp[del_idx] = 1   
# latents = latents[np.zeros(latents.shape[0]) == temp]        
# X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(latents)
# plot_latent_space(X_embedded, true_labels,True)
# -------------------------
tnfs = np.load('../../DeepBin/data/CAMI1_L/vamb/tnf.npz')
abundances = np.load('../../DeepBin/data/CAMI1_L/vamb/rpkm.npz')
tnfs = tnfs['arr_0']
abundances = abundances['arr_0']
# print(abundances.shape)

#------------------------------
# input_data = tnfs
# minmax_scalar = preprocessing.MinMaxScaler()
# input_data = minmax_scalar.fit_transform(input_data)
# tnfs = np.load('../data/tnfs.npy')
# tnfs = tnfs[:16523]
#-----------------------------
zscore(tnfs, axis=0, inplace=True)
zscore(abundances, axis=0, inplace=True)
# -----------------------------
input_data = np.concatenate([tnfs, abundances], axis=1)
# minmax_scalar = preprocessing.MinMaxScaler()
# input_data = minmax_scalar.fit_transform(input_data)
# -----------------------------

# minmax_scalar = preprocessing.MinMaxScaler()
# tnfs = minmax_scalar.fit_transform(tnfs)

# abundances = minmax_scalar.fit_transform(abundances)
# input_data = np.concatenate([tnfs, abundances], axis=1)
train_set = []
test_set = []
contigs = np.load('../../DeepBin/data/CAMI1_L/vamb/contignames.npz')
contigs = contigs['arr_0']
graph = torch.load('../data/CAMI1_L/ag_dict.pkl')
print(graph[1000])
# with open('../data/truedata/metaspades_illumina.fasta', 'rb') as f:
#     contigs = pickle.load(f)
# with open('../data/truedata/metaspades_illumina.fasta', 'r') as f:
#     first_line = True
#     contigs = []
#     for l in f.readlines():
#         if first_line:
#             contigs.append(l[1:-1])
#             first_line = False
#             continue
#         if l.startswith('>'):
#             contigs.append(l[1:-1])
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
    node = int(contigname.split('_')[1])
    new_idx = []
    new_weight = []
    if node in graph.keys():
        graph_nodes = graph[node]
    else:
        similar_m.append(new_idx)
        weight_m.append(new_weight)
        continue
    for i, weight in zip(idx_list, weight_list):
        # print(i)
        for num in graph_nodes:
            if (init_data[i] == node_num[num]).sum() == 104:
                # print(1)
                new_idx.append(i)
                new_weight.append(weight)
                break
    # if len(new_weight) != 0:
    #     m = nn.Softmax(dim=0)
    #     new_weight = m(torch.stack(new_weight))
    #     new_idx = torch.stack(new_idx)
    # print(type(new_idx))
    similar_m.append(new_idx)
    weight_m.append(new_weight)       
    idx += 1
# print(similar_m[:3])
# print(len(similar_m))
# for i in range(100):    
#     print(len(similar_m[i]))
for data, idx_list, weight_list, contigname in zip(init_data, similar_m, weight_m, contignames):
    node = int(contigname.split('_')[1])
    if len(idx_list) == 0:
        train_set.append((data, data, torch.tensor(0.5), contigname))
    else:
        # print(1)
        for idx, weight in zip(idx_list, weight_list):
            # print(idx)
            train_set.append((data, init_data[idx], weight, contigname))
    # if node in graph.keys():
    #     for num in graph[node]:
    #         w = torch.tensor(1)
    #         if torch.cuda.is_available():
    #             w = w.cuda()
    #         train_set.append((data, node_num[num], w, contigname))


# ---------plot initial data-------
# vectors = []
# cons = []
# true_labels = []
# del_idx = []
# for idx, temp in enumerate(train_set):
#     try:
#         data, con = temp
#         true_labels.append(int(label_dic[con]))
#     except KeyError:
#         del_idx.append(idx)
#     vectors.append(data.cpu().detach().numpy())
# true_labels = np.array(true_labels)
# vectors = np.stack(vectors, axis = 0)
# temp = np.zeros(vectors.shape[0]) 
# temp[del_idx] = 1   
# vectors = vectors[np.zeros(vectors.shape[0]) == temp]
# X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(vectors)
# plot_latent_space(X_embedded, true_labels,True)
# ---------------------------------

print(f'After graph embedding: {len(train_set)}')
# print(train_set[2][0].shape)
# print(train_set[7:10])

print(f"length of test_set{len(test_set)}")
# np.save('../data/training_set.npy', input_data)
# with open("../data/sharon/training_set.pkl", "wb") as f:
#     pickle.dump(train_set, f)
# store_large_obj(train_set, '../data/sharon/training_set1.pkl')
torch.save(train_set, '../data/CAMI1_L/training_set.pkl')
torch.save(test_set, '../data/CAMI1_L/test_set.pkl')
# with open("../data/sharon/test_set1.pkl", "wb") as f:
#     pickle.dump(test_set, f)