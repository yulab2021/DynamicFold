import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys
import orjson

data_csv = sys.argv[1]
output_plot = sys.argv[2]

data = pd.read_csv(data_csv)
dynamic_error = list()
static_error = list()
for _, row in data.iterrows():
    reactivity = np.array(orjson.loads(row["RT"]))
    dynamic_prediction = np.array(orjson.loads(row["Predictions"]))
    static_prediction = np.array(orjson.loads(row["RibonanzaNetPredictions"]))
    dynamic_error.extend((reactivity - dynamic_prediction).tolist())
    static_error.extend((reactivity - static_prediction).tolist())

plt.figure(figsize=(5, 5), dpi=300)
sns.kdeplot(dynamic_error, fill=True, label="DynamicFold")
sns.kdeplot(static_error, fill=True, label="RibonanzaNet")
plt.xlabel("Basewise Error")
plt.ylabel("Density")
plt.legend()
plt.title("Distribution of Errors per Base")
plt.savefig(output_plot)
