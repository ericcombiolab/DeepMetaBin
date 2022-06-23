from math import tan
from os import sysconf
import numpy as np
import scipy.stats as stats
from sklearn import preprocessing
import _pickle as pickle
import torch
import sys
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


abundances = np.load('../data/abundances.npy')
tnfs = np.load('../data/tnfs.npy')
print(tnfs.shape)
# input_data = np.concatenate([tnfs, abundances], axis=1)
# input_data = stats.zscore(input_data, axis=1)
minmax_scalar = preprocessing.MinMaxScaler()
tnfs = minmax_scalar.fit_transform(tnfs)
# abundances = minmax_scalar.fit_transform(abundances[:,2:])
abundances = minmax_scalar.fit_transform(abundances)
input_data = np.concatenate([tnfs, abundances], axis=1)

# -----------------------
# input_data = abundances
# -----------------------
train_set = []
contigs = []
with open("../data/contigs.pkl", "rb") as f:
    contigs = pickle.load(f)
for idx, val in enumerate(contigs):
    data = torch.tensor(input_data[idx], dtype=torch.float32)
    # print(data, val, len(contigs), input_data.shape)
    temp = val.split('_')
    if int(temp[3]) > 2000:
        train_set.append((data, val))

print(len(train_set))
print(train_set[:5])
# np.save('../data/training_set.npy', input_data)
with open("../data/training_set.pkl", "wb") as f:
    pickle.dump(train_set, f)