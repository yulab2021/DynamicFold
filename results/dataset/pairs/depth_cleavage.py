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
end_rate = list()
for _, row in dataset.iterrows():
    read_depth.extend(orjson.loads(row["RD"]))
    end_rate.extend(orjson.loads(row["ER"]))

indices = list(range(len(read_depth)))
indices = np.random.choice(indices, size=bootstrap_size, replace=True)
read_depth_sample = np.array(read_depth)[indices]
end_rate_sample = np.array(end_rate)[indices]

plt.figure(figsize=(6, 5), dpi=300)
sns.kdeplot(x=read_depth_sample, y=end_rate_sample, fill=True, levels=levels, cbar=True, cmap=sns.color_palette("Blues", as_cmap=True), gridsize=grid_size)
plt.xlim((0, 1))
plt.ylim((0, 6))
plt.xlabel(r"$\log_{10}$(CPM(Read Depth) + 1)")
plt.ylabel(r"$-\log_{10}$(Cleavage Rate + $10^{-6}$)")
plt.title("Distribution of Read Depth vs. Cleavage Rate per Base")
plt.tight_layout()
plt.savefig(output_plot)
