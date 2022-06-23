import pickle
import numpy as np
import torch

graph = dict()
# with open('../../DeepBin/data/hlj10x/graph.pe', 'r') as f:
#     valid = False
#     key = -1
#     for l in f.readlines():
#         if l.endswith(':\n'):
#             temp = l.split()
#             if int(temp[2][6:-1]) >= 1000:
#                 valid = True
#                 key = int(temp[0].split('_')[1])
#                 if key in graph.keys():
#                     print('Dupilicate keys')
#                 graph[key] = set()
#             else:
#                 valid = False
#         elif valid and l.endswith(';\n'):
#             temp = l.split()
#             if int(temp[2][6:]) >= 1000:
#                 graph[key].add(int(l.split()[0].split('_')[1]))

with open('../data/sharon/graph.ag', 'r') as f:
    valid = False
    key = -1
    for l in f.readlines():
        if l.endswith(':\n'):
            temp = l.split()
            key = int(temp[0].split('_')[1])
            if int(temp[2][6:-1]) >= 1000:
                valid = True
                if key not in graph.keys():
                    graph[key] = set()
            else:
                valid = False
        elif valid and l.endswith(';\n'):
            temp = l.split()
            if int(temp[2][6:]) >= 1000:
                graph[key].add(int(l.split()[0].split('_')[1]))

# print(graph[725])
cnt = 0
for key, val in graph.items():
    cnt += 1
    for i in val:
        cnt += 1
print(cnt)
torch.save(graph, '../data/sharon/ag_dict.pkl')
