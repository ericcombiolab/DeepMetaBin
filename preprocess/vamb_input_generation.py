from math import tan
from os import sysconf
import numpy as np
import scipy.stats as stats
from sklearn import preprocessing
import _pickle as pickle
import torch
import sys

abundances = np.load('../data/abundances.npy')
tnfs = np.load('../data/tnfs.npy')
abundances = np.float32(abundances[:16523])
tnfs = np.float32(tnfs[:16523])
np.savez('../data/tnfs.npz', tnfs)
np.savez('../data/abundances.npz', abundances)
print(tnfs.shape)
# input_data = np.concatenate([tnfs, abundances], axis=1)
# input_data = stats.zscore(input_data, axis=1)
# minmax_scalar = preprocessing.MinMaxScaler()
# tnfs = minmax_scalar.fit_transform(tnfs)
# # abundances = minmax_scalar.fit_transform(abundances[:,2:])
# abundances = minmax_scalar.fit_transform(abundances)
# input_data = np.concatenate([tnfs, abundances], axis=1)
# train_set = []
contigs = []
lengths = []
with open("../data/contigs.pkl", "rb") as f:
    contig_list = pickle.load(f)
    for idx, val in enumerate(contig_list):
        # print(data, val, len(contigs), input_data.shape)
        temp = val.split('_')
        if int(temp[3]) >= 100:
            contigs.append(val)
            lengths.append(int(temp[3]))

contigs = np.array(contigs)
lengths = np.array(lengths)
np.savez('../data/contigs.npz', contigs)
np.savez('../data/lengths.npz', lengths)

# print(len(train_set))
# print(train_set[:5])
# # np.save('../data/training_set.npy', input_data)
# with open("../data/training_set.pkl", "wb") as f:
#     pickle.dump(train_set, f)