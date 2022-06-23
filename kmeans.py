import numpy as np
from scipy.stats.stats import kendalltau
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics.cluster import adjusted_rand_score
import _pickle as pickle
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
import sys

def calculate_precision(predicts, ground_truth):
    predicts_dict = {}
    for clusterNo, contig in predicts:
        if clusterNo in predicts_dict.keys():
            predicts_dict[clusterNo].append(contig)
        else:
            predicts_dict[clusterNo] = [contig]

    ground_truth_dict = {}
    for contig, label in ground_truth:
        ground_truth_dict[contig] = label
    precision_dict = {}
    for key, value in predicts_dict.items():
        precision_dict[key] = {}
        for contig in value:
            if contig not in ground_truth_dict.keys():
                continue
            if ground_truth_dict[contig] in precision_dict[key].keys():
                precision_dict[key][ground_truth_dict[contig]] += 1
            else:
                precision_dict[key][ground_truth_dict[contig]] = 1
    correct_predicts = 0
    total_predicts = 0
    for label_dict in precision_dict.values():
        if len(label_dict.values()) != 0:
            correct_predicts += max(label_dict.values())
            total_predicts += sum(label_dict.values())
    return correct_predicts / total_predicts

      

def calculate_recall(predicts, ground_truth):
    predicts_dict = {}
    for clusterNo, contig in predicts:
        predicts_dict[contig] = clusterNo
    ground_truth_dict = {}
    for contig, label in ground_truth:
        if label in ground_truth_dict.keys():
            ground_truth_dict[label].append(contig)
        else:
            ground_truth_dict[label] = [contig]

    recall_dict = {}
    for key, value in ground_truth_dict.items():
        recall_dict[key] = {}
        for contig in value:
            if contig not in predicts_dict.keys():
                continue
            if predicts_dict[contig] in recall_dict[key].keys():
                recall_dict[key][predicts_dict[contig]] += 1
            else:
                recall_dict[key][predicts_dict[contig]] = 1
    correct_recalls = 0
    total_recalls = 0
    for cluster_dict in recall_dict.values():
        if len(cluster_dict.values()) != 0:
            correct_recalls += max(cluster_dict.values())
            total_recalls += sum(cluster_dict.values())
    return correct_recalls / total_recalls


def calculate_ari(predicts, ground_truth):
    ground_truth_dict = {}
    for contig, label in ground_truth:
        ground_truth_dict[contig] = label
    clusters = []
    labels_true = []
    for clusterNo, contig in predicts:
        if contig not in ground_truth_dict.keys():
            continue
        clusters.append(clusterNo)
        labels_true.append(ground_truth_dict[contig])
    return adjusted_rand_score(clusters, labels_true)

def calculate_accuracy(predicts, ground_truth):
    precision = calculate_precision(predicts, ground_truth)
    recall = calculate_recall(predicts, ground_truth)
    f1_score = 2 * (precision * recall) / (precision + recall)
    ari = calculate_ari(predicts, ground_truth)
    return precision, recall, f1_score, ari

#########################################################
## Data Partition
#########################################################
def partition_dataset(n, proportion=0.8):
  train_num = int(n * proportion)
  indices = np.random.permutation(n)
  train_indices, val_indices = indices[:train_num], indices[train_num:]
  return train_indices, val_indices

train_dataset = []
with open("data/training_set.pkl", "rb") as f:
  train_dataset = pickle.load(f)

train_indices, val_indices = partition_dataset(len(train_dataset), 0.8)
# Create data loaders for train, validation and test datasets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200, sampler=SubsetRandomSampler(train_indices))
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200, sampler=SubsetRandomSampler(val_indices))
train_set = []
train_contigs = []
X = []

for data, contigs in train_loader:
    data = data.view(data.size(0), -1)
    for val in data:
        X.append(val)

X = torch.stack(X,0)
X = X.cpu().detach().numpy()


kmeans_model = KMeans(n_clusters=200, random_state=0).fit(X=X.astype('float'))

test_data = []
test_contigs = []
for data, contigs in test_loader:
    data = data.view(data.size(0), -1)
    for val,contig in zip(data,contigs):
        test_data.append(val.cpu().detach().numpy())
        test_contigs.append(contig)

test_result = kmeans_model.predict(test_data)
predicts = []
for cluster, contig in zip(test_result, test_contigs):
    predicts.append((cluster, contig))

ground_truth = []
with open('data/alignment_result.tsv', 'r') as f:
    for l in f.readlines():
        items = l.split()
        if len(items) == 3:
            continue
        temp = items[0].split('_')
        if int(temp[3]) > 1000:
            ground_truth.append((items[0], items[1]))
precision, recall, f1_score, ari = calculate_accuracy(predicts, ground_truth)
print(ground_truth[:5])

print("Valid - Precision: {:.5f}; Recall: {:.5f}; F1_score: {:.5f}; ARI: {:.5f}".format(precision, recall, f1_score, ari))

