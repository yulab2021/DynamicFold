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
static_error = list()
for _, row in data.iterrows():
    reactivity = np.array(orjson.loads(row["RT"]))
    dynamic_prediction = np.array(orjson.loads(row["Predictions"]))
    static_prediction = np.array(orjson.loads(row["RibonanzaNetPredictions"]))
    dynamic_error.extend((reactivity - dynamic_prediction).tolist())
    static_error.extend((reactivity - static_prediction).tolist())

indices = list(range(len(dynamic_error)))
indices = np.random.choice(indices, size=bootstrap_size, replace=True)
dynamic_sample = np.array(dynamic_error)[indices]
static_sample = np.array(static_error)[indices]

plt.figure(figsize=(6, 5), dpi=300)
sns.kdeplot(x=static_sample, y=dynamic_sample, fill=True, levels=levels, cbar=True, cmap=sns.color_palette("Blues", as_cmap=True), gridsize=grid_size)
plt.xlim((-1, 1))
plt.ylim((-1, 1))
plt.xlabel("RibonanzaNet Error")
plt.ylabel("DynamicFold Error")
plt.title("Distribution of RibonanzaNet vs. DynamicFold Error per Base")
plt.savefig(output_plot)
