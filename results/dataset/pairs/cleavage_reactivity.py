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
reactivity = list()
end_rate = list()
for _, row in dataset.iterrows():
    reactivity.extend(orjson.loads(row["RT"]))
    end_rate.extend(orjson.loads(row["ER"]))

indices = list(range(len(reactivity)))
indices = np.random.choice(indices, size=bootstrap_size, replace=True)
reactivity_sample = np.array(reactivity)[indices]
end_rate_sample = np.log10(np.array(end_rate)[indices] + eps)

plt.figure(figsize=(6, 5), dpi=300)
sns.kdeplot(x=end_rate_sample, y=reactivity_sample, fill=True, levels=levels, cbar=True, cmap=sns.color_palette("Blues", as_cmap=True), gridsize=grid_size)
plt.xlim((np.log10(eps), 0))
plt.ylim((0, 1))
plt.xlabel(f"$\\log_{{10}}$(Cleavage Rate + {eps})")
plt.ylabel("Reactivity Score")
plt.title("Distribution of Cleavage Rate vs. Reactivity Score per Base")
plt.savefig(output_plot)
