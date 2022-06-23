import math
import timeit
import numpy as np
import _pickle as pickle

class Tree_Node():
    def __init__(self, base):
        self.base = base
        self.A_node = None
        self.T_node = None
        self.C_node = None
        self.G_node = None

    def __repr__(self):
        return self.base

class Kmers_tree():
    def __init__(self, k):
        self.root = Tree_Node('')
        self.k = k
        self.all_kmers = list()


    def generate_tree(self, node, k):
        if k > 0:
            node.A_node = Tree_Node('A')
            node.T_node = Tree_Node('T')
            node.C_node = Tree_Node('C')
            node.G_node = Tree_Node('G')
            self.generate_tree(node.A_node, k - 1)
            self.generate_tree(node.T_node, k - 1)
            self.generate_tree(node.C_node, k - 1)
            self.generate_tree(node.G_node, k - 1)

    def toList(self):
        self.generate_tree(self.root, self.k)
        self.recursive_toList(self.root,"")
        return self.all_kmers

    def recursive_toList(self, node, prev_str):
        prev_str = prev_str + node.base
        if node.A_node is not None:
            self.recursive_toList(node.A_node, prev_str)
            self.recursive_toList(node.T_node, prev_str)
            self.recursive_toList(node.C_node, prev_str)
            self.recursive_toList(node.G_node, prev_str)
        else:
            self.all_kmers.append(prev_str)

class Sliding_window():
    def __init__(self, iter_obj, window_size, stride):
        self.window_size = window_size
        self.stride = stride
        self.iter_obj = iter_obj
        self.left_p = 0

    def forward(self):
        if self.left_p + 4 > len(self.iter_obj):
            return False
        kmer = ''
        cnt = 0
        for idx, val in enumerate(self.iter_obj[self.left_p:], self.left_p):
            cnt += 1
            if not (val == 'A' or val == 'T' or val == 'C' or val == 'G'):
                self.left_p = idx + 1
                return 'N'
            else:
                kmer += val
            if cnt == self.window_size:
                self.left_p += 1
                break
        return kmer

    def hasNext(self):
        return self.left_p + self.window_size <= len(self.iter_obj)

def revComp(encoded_x, k):
    decoding_scheme = {encoding_scheme['A']: 'A', encoding_scheme['T']: 'T',
                       encoding_scheme['C']:'C', encoding_scheme['G']: 'G'}
    mask = int(math.pow(2, k * 2) - 1)
    base_mask = int(math.pow(2, k * 2 - 1) + math.pow(2, k * 2 - 2))
    val = 0
    for i in range(k):
        val <<= 2
        encoded_base = (encoded_x & base_mask) >> ((k - i - 1) * 2)
        if decoding_scheme[encoded_base] == 'A':
            val += encoding_scheme['T']
        elif decoding_scheme[encoded_base] == 'T':
            val += encoding_scheme['A']
        elif decoding_scheme[encoded_base] == 'C':
            val += encoding_scheme['G']
        elif decoding_scheme[encoded_base] == 'G':
            val += encoding_scheme['C']
        base_mask >>= 2
    return val & mask

def bit_wise_encode(kmers_list, k):
    kmer2freq = dict()
    mask = int(math.pow(2, k * 2) - 1)
    val = 0
    for seq in kmers_list:
        val >>= (k * 2)
        for base in seq:
            val = val << 2
            val += encoding_scheme[base]
        val = val & mask
        kmer2freq[min(val, revComp(val, k))] = 0
    return kmer2freq

def generate_all_kmers(k):
    kmer_tree = Kmers_tree(k)
    return kmer_tree.toList()

def count_kmers(k, contig, kmer2freq):
    mask = int(math.pow(2, k * 2) - 1)
    sliding_window = Sliding_window(contig, k, 1)
    kmer = sliding_window.forward()
    val = encode(kmer, k)
    kmer2freq[min(val, revComp(val, k))] += 1
    while sliding_window.hasNext():
        kmer = sliding_window.forward()
        accumulate = False
        if kmer == 'N':
            accumulate = True
            continue
        elif accumulate:
            accumulate = False
            val = encode(kmer, k)
            kmer2freq[min(val, revComp(val, k))] += 1
        else:
            val = (val << 2) & mask
            val += encoding_scheme[kmer[-1]]
            kmer2freq[min(val, revComp(val, k))] += 1
    # kmer_sum = 0
    # for val in kmer2freq.values():
    #     kmer_sum += val
    # for key, value in kmer2freq.items():
    #     kmer2freq[key] = value / kmer_sum

def encode(kmer, k):
    mask = int(math.pow(2, k * 2) - 1)
    val = 0
    for base in kmer:
        val = val << 2
        val += encoding_scheme[base]
    return val & mask


if __name__ == '__main__':
    start_time = timeit.default_timer()
    global encoding_scheme, output_vectors
    k = 4
    output_vectors = list()
    encoding_scheme = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    all_kmers = generate_all_kmers(k)
    kmer2freq = bit_wise_encode(all_kmers, k)
    with open('../data/contigs.fasta', 'r') as f:
        i = 1
        first_line = True
        contig_name = ''
        contig = ''
        contigs = []
        for l in f.readlines():
            if first_line:
                contigs.append(l[1:-1])
                contig_name = l
                first_line = False
                continue
            if l.startswith('>'):
                count_kmers(k, contig, kmer2freq)
                output_vector = np.array(list(kmer2freq.values()))
                output_vectors.append(output_vector)
                print(f'{contig_name}\n{kmer2freq}')
                for group in kmer2freq.keys():
                    kmer2freq[group] = 0
                contigs.append(l[1:-1])
                contig = ''
                contig_name = l
            else:
                contig += l[:-1]
        count_kmers(k, contig, kmer2freq)
        output_vector = np.array(list(kmer2freq.values()))
        output_vectors.append(output_vector)
        output_vectors = np.stack(output_vectors, axis=0)
        print(f'{contig_name}\n{kmer2freq}')
        np.save('../data/tnfs.npy', output_vectors)
        print(contigs[:5])
        with open("../data/contigs.pkl", "wb") as f:
            pickle.dump(contigs, f)
        print('time: {:.3f}s'.format(timeit.default_timer() - start_time))




