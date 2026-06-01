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
end_rate = list()
mismatch_rate = list()
for _, row in dataset.iterrows():
    end_rate.extend(orjson.loads(row["ER"]))
    mismatch_rate.extend(orjson.loads(row["MR"]))

indices = list(range(len(end_rate)))
indices = np.random.choice(indices, size=bootstrap_size, replace=True)
end_rate_sample = np.array(end_rate)[indices]
mismatch_rate_sample = np.array(mismatch_rate)[indices]

plt.figure(figsize=(6, 5), dpi=300)
sns.kdeplot(x=end_rate_sample, y=mismatch_rate_sample, fill=True, levels=levels, cbar=True, cmap=sns.color_palette("Blues", as_cmap=True), gridsize=grid_size)
plt.xlim((0, 6))
plt.ylim((0, 6))
plt.xlabel(r"$-\log_{10}$(Cleavage Rate + $10^{-6}$)")
plt.ylabel(r"$-\log_{10}$(Mismatch Rate + $10^{-6}$)")
plt.title("Distribution of Cleavage Rate vs. Mismatch Rate per Base")
plt.tight_layout()
plt.savefig(output_plot)
