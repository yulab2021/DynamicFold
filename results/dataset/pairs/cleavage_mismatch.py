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
end_rate = list()
mismatch_rate = list()
for _, row in dataset.iterrows():
    end_rate.extend(orjson.loads(row["ER"]))
    mismatch_rate.extend(orjson.loads(row["MR"]))

indices = list(range(len(end_rate)))
indices = np.random.choice(indices, size=bootstrap_size, replace=True)
end_rate_sample = np.log10(np.array(end_rate)[indices] + eps)
mismatch_rate_sample = np.log10(np.array(mismatch_rate)[indices] + eps)

plt.figure(figsize=(6, 5), dpi=300)
sns.kdeplot(x=end_rate_sample, y=mismatch_rate_sample, fill=True, levels=levels, cbar=True, cmap=sns.color_palette("Blues", as_cmap=True), gridsize=grid_size)
plt.xlim((np.log10(eps), 0))
plt.ylim((np.log10(eps), 0))
plt.xlabel(f"$\\log_{{10}}$(Cleavage Rate + {eps})")
plt.ylabel(f"$\\log_{{10}}$(Mismatch Rate + {eps})")
plt.title("Distribution of Cleavage Rate vs. Mismatch Rate per Base")
plt.savefig(output_plot)
