import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import orjson

dataset_csv = sys.argv[1]
bootstrap_size = int(sys.argv[2])
eps = float(sys.argv[3])
levels = int(sys.argv[4])
grid_size = int(sys.argv[5])
output_plot = sys.argv[6]

dataset = pd.read_csv(dataset_csv)
read_depth = list()
mismatch_rate = list()
for _, row in dataset.iterrows():
    read_depth.extend(orjson.loads(row["RD"]))
    mismatch_rate.extend(orjson.loads(row["MR"]))

indices = list(range(len(read_depth)))
indices = np.random.choice(indices, size=bootstrap_size, replace=True)
read_depth_sample = np.array(read_depth)[indices]
mismatch_rate_sample = np.log10(np.array(mismatch_rate)[indices] + eps)

plt.figure(figsize=(6, 5), dpi=300)
sns.kdeplot(x=read_depth_sample, y=mismatch_rate_sample, fill=True, levels=levels, cbar=True, cmap=sns.color_palette("Blues", as_cmap=True), gridsize=grid_size)
plt.xlim((0, 1))
plt.ylim((np.log10(eps), 0))
plt.xlabel("Winsorize-Scaled Read Depth")
plt.ylabel(f"$\\log_{{10}}$(Mismatch Rate + {eps})")
plt.title("Distribution of Read Depth vs. Mismatch Rate per Base")
plt.savefig(output_plot)
