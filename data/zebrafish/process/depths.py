# depths.py

import pandas as pd
import sys
import utils
import multiprocessing as mp
from tqdm import tqdm

label_csv = sys.argv[1]
metrics_db = sys.argv[2]
rtstops_db = sys.argv[3]
batch_size = int(sys.argv[4])
output_csv = sys.argv[5]

def count(key):
    entry = database.read(key)
    counts = sum(entry["ED"])
    srr = key.split('|')[3]
    return srr, counts

labels = utils.Label(label_csv)
depths = {srr: 0 for srr in labels.get_srr_list()}

metrics = utils.Database(metrics_db, "metrics")
metrics.connect()
rtstops = utils.Database(rtstops_db, "rtstops")
rtstops.connect()

database = metrics
with mp.Pool(processes=batch_size) as pool:
    for srr, counts in tqdm(pool.imap_unordered(count, metrics.list()), total=len(metrics.list()), desc="Count Metrics"):
        depths[srr] += counts

database = rtstops
with mp.Pool(processes=batch_size) as pool:
    for srr, counts in tqdm(pool.imap_unordered(count, rtstops.list()), total=len(rtstops.list()), desc="Count RTStops"):
        depths[srr] += counts

depths = pd.Series(depths)
depths.name = "depth"
depths.index.name = "SRR"
depths.to_csv(output_csv)
