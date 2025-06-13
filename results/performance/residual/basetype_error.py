import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import orjson

data_csv = sys.argv[1]
output_plot = sys.argv[2]

data = pd.read_csv(data_csv)
error = {"A": list(), "C": list(), "G": list(), "U": list()}
for _, row in data.iterrows():
    seq = row["SQ"]
    reactivity = np.array(orjson.loads(row["RT"]))
    prediction = np.array(orjson.loads(row["Predictions"]))
    entry_error = reactivity - prediction
    for s, e in zip(list(seq), entry_error.tolist()):
        error[s].append(e)

plt.figure(figsize=(5, 5), dpi=300)
plt.violinplot(list(error.values()), positions=list(range(len(error))), widths=0.8, showmeans=True)
plt.xticks(range(len(error)), list(error.keys()))
plt.title("Distribution of Error per Base")
plt.savefig(output_plot)
