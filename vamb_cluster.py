import vamb
import time
import numpy as np
import _pickle as pickle 


def cluster(clusterspath, latent, contignames, windowsize=200, minsuccesses=20, separator=None, cuda=False):
    begintime = time.time()

    it = vamb.cluster.cluster(latent, contignames, destroy=True, windowsize=windowsize,
                              normalized=False, minsuccesses=minsuccesses, cuda=cuda)

    renamed = ((str(i+1), c) for (i, (n,c)) in enumerate(it))

    # Binsplit if given a separator
    if separator is not None:
        renamed = vamb.vambtools.binsplit(renamed, separator)

    with open(clusterspath, 'w') as clustersfile:
        _ = vamb.vambtools.write_clusters(clustersfile, renamed, max_clusters=20,
                                          min_size=1, rename=False)
    clusternumber, ncontigs = _

    elapsed = round(time.time() - begintime, 2)

latents = np.load('data/latents.npy')
clusterspath = 'data/outdata1.tsv'
contignames = []
with open("data/contignames.pkl", "rb") as f:
    contignames = pickle.load(f)
cluster(clusterspath, latents, contignames)

    




