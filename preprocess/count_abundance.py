import os
import subprocess
import math
import _pickle as pickle
import numpy as np
import timeit

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

def revComp(encoded_x, k=15):
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


def count_kmers(contig, mapper,output_vector, k=15):
    mask = int(math.pow(2, k * 2) - 1)
    sliding_window = Sliding_window(contig, k, 1)
    error_kmer = 0
    while True:
        kmer = sliding_window.forward()
        if kmer != 'N':
            break
    val = encode(kmer, k)
    try:
        if val in mapper.keys():
            output_vector[mapper[val] // interval] += 1
        else:
            output_vector[mapper[revComp(val)] // interval] += 1
    except KeyError:
        error_kmer += 1

    while sliding_window.hasNext():
        kmer = sliding_window.forward()
        accumulate = False
        if kmer == 'N':
            accumulate = True
            continue
        elif accumulate:
            accumulate = False
            val = encode(kmer, k)
        else:
            val = (val << 2) & mask
            val += encoding_scheme[kmer[-1]]
        try:
            if val in mapper.keys():
                output_vector[mapper[val] // interval] += 1
            else:
                output_vector[mapper[revComp(val)] // interval] += 1
        except KeyError:
            error_kmer += 1
    return error_kmer


def encode(kmer, k=15):
    mask = int(math.pow(2, k * 2) - 1)
    val = 0
    for base in kmer:
        val = val << 2
        val += encoding_scheme[base]
    return val & mask

def run_jellyfish():
    print(
        f'jellyfish_count starts: {" ".join(["pigz", "-dc", "SRR8359173.1_1.fastq.gz", "SRR8359173.1_2.fastq.gz", "|", "jellyfish", "count", "-t", "100", "-C", "-m", "15", "-s", "5G", "-o", "jellyfish_count", "--min-qual-char=?", "/dev/fd/0"])}',
        "CMD",
    )
    pipe = subprocess.Popen(
        ["pigz", "-dc", "SRR8359173.1_1.fastq.gz", "SRR8359173.1_2.fastq.gz"], stdout=subprocess.PIPE
    )
    subprocess.check_output(
        [
            "jellyfish",
            "count",
            "-t",
            "100",
            "-C",
            "-m",
            "15",
            "-s",
            "5G",
            "-o",
            "jellyfish_count",
            "--min-qual-char=?",
            "/dev/fd/0",
        ],
        stdin=pipe.stdout,
    )
    pipe.communicate()
    print(
        f'jellyfish_count ends: {" ".join(["pigz", "-dc", "SRR8359173.1_1.fastq.gz", "SRR8359173.1_1.fastq.gz", "|", "jellyfish", "count", "-t", "100", "-C", "-m", "15", "-s", "5G", "-o", "jellyfish_count", "--min-qual-char=?", "/dev/fd/0"])}',
        "CMD",
    )
    print(
        f'jellyfish_dump start: {" ".join(["jellyfish","dump","-c","-t","jellyfish_count","-o","jellyfish_dump"])}',
        "CMD",
    )
    pipe = subprocess.Popen(
        [
            "jellyfish",
            "dump",
            "-c",
            "-t",
            "jellyfish_count",
            "-o",
            "jellyfish_dump",
        ],
    )
    pipe.wait()
    print(
        f'jellyfish_dump end: {" ".join(["jellyfish", "dump", "-c", "-t", "jellyfish_count", "-o", "jellyfish_dump"])}',
        "CMD",
    )
    print('Encoding jellyfish_dump')
    mapper = dict()
    with open('jellyfish_dump', 'r') as f:
        for l in f.readlines():
            out = l.split()
            kmer = out[0]
            cnt = int(out[1])
            mapper[encode(kmer)] = cnt

    with open('jellyfish_dump_binary.pkl', 'wb') as f:
        pickle.dump(mapper, f)
        print('Finish storing the mapper')
    return mapper


if __name__ == '__main__':
    start_time = timeit.default_timer()
    global encoding_scheme
    mapper = dict()
    encoding_scheme = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    if not os.path.isfile("../data/jellyfish_dump_binary.pkl"):
        mapper = run_jellyfish()
    else:
        with open('../data/jellyfish_dump_binary.pkl', "rb") as f:
            mapper = pickle.load(f)
            print('Finish load the mapper')

    '''
         we assume the parameter m = max(mapper.values) (i.e. 1) n = min(mapper.values) (i.e. 8000) interval = 100
         Now, we will generate a vector with size 8000/100 = 80

    '''
    global vector_size, interval, output_vectors
    interval = 100
    vector_size = 80
    output_vectors = list()

    with open('../data/contigs.fasta', 'r') as f:
        first_line = True
        contig_name = ''
        contig = ''
        output_vector = np.zeros(vector_size, dtype=int)
        contigs = []
        for l in f.readlines():
            if first_line:
                contigs.append(l[1:-1])
                contig_name = l
                first_line = False
                continue
            if l.startswith('>'):
                error_kmer = count_kmers(contig, mapper,output_vector)
                print(f'{contig_name} (miss: {error_kmer})\n{output_vector}')
                output_vectors.append(output_vector)
                output_vector = np.zeros(vector_size, dtype=int)
                contigs.append(l[1:-1])
                contig = ''
                contig_name = l
            else:
                contig += l[:-1]
        error_kmer = count_kmers(contig, mapper, output_vector)
        print(f'{contig_name} (miss: {error_kmer})\n{output_vector}')
        output_vectors.append(output_vector)
        output_vectors = np.stack(output_vectors, axis=0)
        # output_vectors = output_vectors[1:]
        np.save('../data/abundances.npy', output_vectors)
        with open("../data/contigs.pkl", "wb") as f:
            pickle.dump(contigs, f)
        print('time: {:.3f}s'.format(timeit.default_timer() - start_time))


