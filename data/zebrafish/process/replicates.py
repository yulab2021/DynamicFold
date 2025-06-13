# replicates.py

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import utils
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

database_db = sys.argv[1]
table_name = sys.argv[2]
annotations_csv = sys.argv[3]
output_dir = sys.argv[4]
batch_size = int(sys.argv[5])

def arrange_keys(keys):
    key_maps = dict()
    srr_maps = dict()
    for key in keys:
        sample, experiment, group, srr, ref_name = key.split("|")
        legend = f"{sample}|{experiment}|{group}"
        if srr not in key_maps:
            key_maps[srr] = dict()
        if legend not in srr_maps:
            srr_maps[legend] = list()
        if srr not in srr_maps[legend]:
            srr_maps[legend].append(srr)
        key_maps[srr][ref_name] = key
    
    return key_maps, srr_maps

def load_data(args):
    srr, ref_names = args
    cache = dict()
    data = {ref_name: 0 for ref_name in annotations.index}
    total_reads = 0
    for ref_name, key in ref_names.items():
        entry_data = database.read(key)
        entry_reads = np.sum(entry_data["ED"])
        entry_length = annotations.loc[ref_name, "length"]
        total_reads += entry_reads
        cache[ref_name] = entry_reads / entry_length
    for ref_name, entry_data in cache.items():
        data[ref_name] = entry_data / total_reads
    return srr, data

def group_points(data, srr_maps):
    data_grouped = dict()
    for legend, srr_list in srr_maps.items():
        data_grouped[legend] = data.loc[srr_list]
    return data_grouped

# Load data
annotations = pd.read_csv(annotations_csv)
annotations = annotations.loc[annotations.groupby("Parent")["length"].idxmax()].set_index("id").sort_index()
database = utils.Database(database_db, table_name)
database.connect()
keys = database.list()
key_maps, srr_maps = arrange_keys(keys)
data = dict()

with mp.Pool(processes=batch_size) as pool:
    for srr, srr_data in tqdm(pool.imap_unordered(load_data, key_maps.items()), total=len(key_maps), desc="Load Data"):
        data[srr] = srr_data

database.close()
data = pd.DataFrame(data).transpose()

# PCA decomposition
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
data_pca = pd.DataFrame(data_pca, index=data.index, columns=['Component 1', 'Component 2'])

# Create scatter plot
data_grouped = group_points(data_pca, srr_maps)
plt.figure(figsize=(10, 10), dpi=300)
for legend, data in data_grouped.items():
    plt.scatter(data.iloc[:,0], data.iloc[:,1], label=legend)
for index in data_pca.index:
    plt.annotate(index, (data_pca.loc[index, 'Component 1'], data_pca.loc[index, 'Component 2']))
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.title(table_name)
plt.savefig(f"{output_dir}/{table_name}_replicates.png")
