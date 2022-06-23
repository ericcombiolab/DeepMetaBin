import _pickle as pickle
import gzip
import numpy as np
with open('../data/truedata/metaspades_illumina.fasta', 'r') as f:
    first_line = True
    contigs = []
    for l in f.readlines():
        if first_line:
            contigs.append(l[1:-1])
            first_line = False
            continue
        if l.startswith('>'):
            contigs.append(l[1:-1])

print(len(contigs))
# print(contigs[-5:])
contigs = np.array(contigs)
np.savez('../data/truedata/contigs.npz', contigs)
# with open("../data/truedata/contigs.pkl", "wb") as f:
#     pickle.dump(contigs, f)