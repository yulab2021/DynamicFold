import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import orjson

dataset_csv = sys.argv[1]
bootstrap_size = int(sys.argv[2])
levels = int(sys.argv[3])
grid_size = int(sys.argv[4])
output_plot = sys.argv[5]

dataset = pd.read_csv(dataset_csv)
read_depth = list()
reactivity = list()
for _, row in dataset.iterrows():
    read_depth.extend(orjson.loads(row["RD"]))
    reactivity.extend(orjson.loads(row["RT"]))

indices = list(range(len(reactivity)))
indices = np.random.choice(indices, size=bootstrap_size, replace=True)
read_depth_sample = np.array(read_depth)[indices]
reactivity_sample = np.array(reactivity)[indices]

plt.figure(figsize=(6, 5), dpi=300)
sns.kdeplot(x=read_depth_sample, y=reactivity_sample, fill=True, levels=levels, cbar=True, cmap=sns.color_palette("Blues", as_cmap=True), gridsize=grid_size)
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.xlabel("Winsorize-Scaled Read Depth")
plt.ylabel("Reactivity Score")
plt.title("Distribution of Read Depth vs. Reactivity Score per Base")
plt.savefig(output_plot)
