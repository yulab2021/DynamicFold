import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import orjson

dataset_csv = sys.argv[1]
bootstrap_size = int(sys.argv[2])
output_plot = sys.argv[3]

dataset = pd.read_csv(dataset_csv)
prediction = list()
reactivity = list()
for _, row in dataset.iterrows():
    prediction.extend(orjson.loads(row["RibonanzaNetPredictions"]))
    reactivity.extend(orjson.loads(row["RT"]))

indices = list(range(len(reactivity)))
indices = np.random.choice(indices, size=bootstrap_size, replace=True)
prediction_sample = np.array(prediction)[indices]
reactivity_sample = np.array(reactivity)[indices]
ref_line = np.arange(0, 1, 0.01)

plt.figure(figsize=(5, 5), dpi=300)
plt.scatter(prediction_sample, reactivity_sample, s=0.1, marker="o")
plt.plot(ref_line, ref_line, color='black', linestyle='--', alpha=0.8)
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.xlabel("RibonanzaNet Prediction")
plt.ylabel("Reactivity Score")
plt.title("Distribution of RibonanzaNet Prediction vs. Reactivity Score per Base")
plt.savefig(output_plot)
