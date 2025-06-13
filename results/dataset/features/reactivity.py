import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import orjson

dataset_csv = sys.argv[1]
output_plot = sys.argv[2]

dataset = pd.read_csv(dataset_csv)
experimental = list()
predicted = list()
for _, row in dataset.iterrows():
    experimental.extend(orjson.loads(row["RT"]))
    predicted.extend(orjson.loads(row["RibonanzaNetPredictions"]))

plt.figure(figsize=(5, 5), dpi=300)
sns.kdeplot(experimental, fill=True, label="Experimental")
sns.kdeplot(predicted, fill=True, label="RibonanzaNet")
plt.xlabel("Reactivity Score")
plt.ylabel("Density")
plt.legend()
plt.title("Distribution of icSHAPE Reactivity per Base")
plt.savefig(output_plot)
