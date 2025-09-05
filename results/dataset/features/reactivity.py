import sys
import matplotlib.pyplot as plt
import pandas as pd
import orjson

dataset_csv = sys.argv[1]
num_bins = int(sys.argv[2])
output_plot = sys.argv[3]

dataset = pd.read_csv(dataset_csv)
reactivity = list()
for _, row in dataset.iterrows():
    reactivity.extend(orjson.loads(row["RT"]))

plt.figure(figsize=(5, 5), dpi=300)
plt.hist(reactivity, bins=num_bins, edgecolor="white")
plt.xlabel("Reactivity Score")
plt.ylabel("Frequency")
plt.title("Distribution of icSHAPE Reactivity per Base")
plt.tight_layout()
plt.savefig(output_plot)
