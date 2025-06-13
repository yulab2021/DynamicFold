import sys
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial

dataset_csv = sys.argv[1]
min_len = int(sys.argv[2])
max_len = int(sys.argv[3])
output_plot = sys.argv[4]
batch_size = int(sys.argv[5])

def add_dict(this, other):
    for key, val in other.items():
        if key in this:
            this[key] += val
        else:
            this[key] = val
    return this

def count_motif(sequence, motif_length):
    end = len(sequence) - motif_length + 1
    motifs = dict()
    for start in range(motif_length):
        for i in range(start, end, motif_length):
            motif = sequence[i:i+motif_length]
            if 'N' in motif:
                continue
            if motif in motifs:
                motifs[motif] += 1
            else:
                motifs[motif] = 1
    return motifs

def summarize_counts(motif_counts):
    completeness = dict()
    for motif_length in range(min_len, max_len + 1):
        observed = len(list(motif_counts[motif_length].values()))
        expected = 4**motif_length
        completeness[motif_length] = observed / expected
    return completeness

dataset = pd.read_csv(dataset_csv)
sequences = dataset["SQ"].to_list()
motif_counts = dict()

with mp.Pool(processes=batch_size) as pool:
    for motif_length in tqdm(range(min_len, max_len + 1), desc=f"Count Motif"):
        partial_count = partial(count_motif, motif_length=motif_length)
        motif_counts[motif_length] = dict()
        for motifs in pool.imap_unordered(partial_count, sequences):
            motif_counts[motif_length] = add_dict(motif_counts[motif_length], motifs)

completeness = summarize_counts(motif_counts)
keys = list(completeness.keys())
values = list(completeness.values())
remainings = [1 - v for v in values]

plt.figure(figsize=(5, 5), dpi=300)
plt.bar(keys, values)
plt.bar(keys, remainings, bottom=values, color='lightgrey')
plt.xlabel("Motif Length")
plt.ylabel("Coverage")
plt.title(f"Sequence Motif Diversity")
plt.savefig(output_plot)
