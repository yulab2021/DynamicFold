import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys
import orjson

data_csv = sys.argv[1]
bootstrap_size = int(sys.argv[2])
levels = int(sys.argv[3])
grid_size = int(sys.argv[4])
output_plot = sys.argv[5]

data = pd.read_csv(data_csv)
dynamic_error = list()
reactivity_score = list()
for _, row in data.iterrows():
    reactivity = np.array(orjson.loads(row["RT"]))
    dynamic_prediction = np.array(orjson.loads(row["Predictions"]))
    dynamic_error.extend((reactivity - dynamic_prediction).tolist())
    reactivity_score.extend(orjson.loads(row["RT"]))

indices = list(range(len(dynamic_error)))
indices = np.random.choice(indices, size=bootstrap_size, replace=True)
reactivity_sample = np.array(reactivity_score)[indices]
error_sample = np.array(dynamic_error)[indices]

plt.figure(figsize=(6, 5), dpi=300)
sns.kdeplot(x=reactivity_sample, y=error_sample, fill=True, levels=levels, cbar=True, cmap=sns.color_palette("Blues", as_cmap=True), gridsize=grid_size)
plt.xlim((0, 1))
plt.ylim((-1, 1))
plt.xlabel("icSHAPE Reactivity")
plt.ylabel("DynamicFold Error")
plt.title("Distribution of True Reactivity vs. Error per Base")
plt.savefig(output_plot)