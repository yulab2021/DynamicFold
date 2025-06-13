import pandas as pd
import matplotlib.pyplot as plt
import sys

dataset_csv = sys.argv[1]
output_plot = sys.argv[2]

dataset = pd.read_csv(dataset_csv)
proportions = {"A": list(), "C": list(), "G": list(), "U": list()}

for seq in dataset["SQ"]:
    seq = str(seq)
    length = len(seq)
    for base in proportions:
        proportions[base].append(seq.count(base) / length)

plt.figure(figsize=(5, 5), dpi=300)
plt.violinplot(list(proportions.values()), positions=list(range(len(proportions))), widths=0.8, showmeans=True)
plt.xticks(range(len(proportions)), list(proportions.keys()))
plt.axhline(y=0.25, color='black', linestyle='--', alpha=0.8, linewidth=1)
plt.ylabel("Proportion")
plt.title("Distribution of Base Proportions per Sequence")
plt.savefig(output_plot)
