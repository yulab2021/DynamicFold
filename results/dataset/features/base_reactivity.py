import pandas as pd
import matplotlib.pyplot as plt
import orjson
import sys

dataset_csv = sys.argv[1]
output_plot = sys.argv[2]

dataset = pd.read_csv(dataset_csv)
reactivity = {"A": list(), "C": list(), "G": list(), "U": list()}

for seq, rt in zip(dataset["SQ"], dataset["RT"]):
    rt = orjson.loads(rt)
    seq = list(seq)
    for s, r in zip(seq, rt):
        if s in reactivity:
            reactivity[s].append(r)

plt.figure(figsize=(5, 5), dpi=300)
plt.violinplot(list(reactivity.values()), positions=list(range(len(reactivity))), widths=0.8, showmeans=True)
plt.xticks(range(len(reactivity)), list(reactivity.keys()))
plt.ylabel("icSHAPE Reactivity")
plt.title("Distribution of icSHAPE Reactivity per Base")
plt.savefig(output_plot)
