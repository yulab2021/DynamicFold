import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import orjson

data_csv = sys.argv[1]
bootstrap_size = int(sys.argv[2])
output_plot = sys.argv[3]

data = pd.read_csv(data_csv)
reactivity = list()
prediction = list()
for _, row in data.iterrows():
    reactivity.extend(orjson.loads(row["RT"]))
    prediction.extend(orjson.loads(row["Predictions"]))
ref_line = np.arange(0, 1, 0.01)

indices = list(range(len(reactivity)))
indices = np.random.choice(indices, size=bootstrap_size, replace=True)
prediction_sample = np.array(prediction)[indices]
reactivity_sample = np.array(reactivity)[indices]

plt.figure(figsize=(5, 5), dpi=300)
plt.scatter(prediction_sample, reactivity_sample, s=0.1, marker="o")
plt.plot(ref_line, ref_line, color='black', linestyle='--', alpha=0.8)
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.xlabel("DynamicFold Prediction")
plt.ylabel("Reactivity Score")
plt.title(f"DynamicFold Prediction vs. True Reactivity")
plt.savefig(output_plot)
