# completeness.py

import sys
import sqlite3
import multiprocessing as mp
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial

database_db = sys.argv[1]
table_name = sys.argv[2]
clause = sys.argv[3]
max_len = int(sys.argv[4])
output_dir = sys.argv[5]
batch_size = int(sys.argv[6])

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
    p_values = dict()
    for motif_length in range(1, max_len + 1):
        observed = len(list(motif_counts[motif_length].values()))
        expected = 4**motif_length
        observed_counts = np.array(list(motif_counts[motif_length].values()))
        expected_counts = observed_counts.sum() / expected
        completeness[motif_length] = observed / expected
        chi_squared = np.sum((observed_counts - expected_counts)**2 / expected_counts) + (expected - observed) * expected_counts
        p_values[motif_length] = stats.chi2.sf(chi_squared, expected - 1)
    return completeness, p_values

conn = sqlite3.connect(database_db)
cursor = conn.cursor()
cursor.execute(f"SELECT Sequence FROM {table_name} {clause}")
rows = cursor.fetchall()
sequences = [row[0] for row in rows]
motif_counts = dict()
conn.close()

with mp.Pool(processes=batch_size) as pool:
    for motif_length in tqdm(range(1, max_len + 1), desc=f"Count Motif"):
        partial_count = partial(count_motif, motif_length=motif_length)
        motif_counts[motif_length] = dict()
        for motifs in pool.imap_unordered(partial_count, sequences):
            motif_counts[motif_length] = add_dict(motif_counts[motif_length], motifs)

completeness, p_values = summarize_counts(motif_counts)

plt.figure(figsize=(10, 5), dpi=300)
plt.plot(list(completeness.keys()), list(completeness.values()))
plt.xlabel("Motif Length")
plt.ylabel("Proportion Complete")
plt.title(f"{table_name}: Completeness")
plt.savefig(f"{output_dir}/{table_name}_completeness.png")

plt.figure(figsize=(10, 5), dpi=300)
plt.plot(list(p_values.keys()), list(p_values.values()))
plt.xlabel("Motif Length")
plt.ylabel("P-Value Balanced")
plt.title(f"{table_name}: Balance")
plt.savefig(f"{output_dir}/{table_name}_balance.png")
