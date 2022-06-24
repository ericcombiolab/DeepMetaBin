import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score

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
    total_recalls = 0
    for key, value in ground_truth_dict.items():
        recall_dict[key] = {}
        for contig in value:
            if contig not in predicts_dict.keys():
                total_recalls += 1
                continue
            if predicts_dict[contig] in recall_dict[key].keys():
                recall_dict[key][predicts_dict[contig]] += 1
            else:
                recall_dict[key][predicts_dict[contig]] = 1
    correct_recalls = 0
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

predicts = []
with open('/tmp/local/zmzhang/12mock/busybeeweb/binning_8c06596b-d11a-4e61-8181-5c66979c3941/contigs.bin.table.tsv', 'r') as f:
    for l in f.readlines():
        items = l.split()
        if len(items) == 3:
            continue
        temp = items[0].split('_')
        if int(temp[3]) >= 1000:
            predicts.append((int(items[1]), items[0]))

ground_truth = []
with open('../GMVAE_Graph_fusion/data/12mock/labels.csv', 'r') as f:
    for l in f.readlines():
        items = l.split(',')
        if len(items) == 3:
            continue
        temp = items[0].split('_')
        if int(temp[3]) >= 1000:
            ground_truth.append((items[0], items[1]))

print(len(predicts))
# print(ground_truth[:5])
precision, recall, f1_score, ari = calculate_accuracy(predicts, ground_truth)
print("Valid - Precision: {:.5f}; Recall: {:.5f}; F1_score: {:.5f}; ARI: {:.5f}".format(precision, recall, f1_score, ari))
