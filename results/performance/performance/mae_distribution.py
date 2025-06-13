import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

data_csv = sys.argv[1]
output_plot = sys.argv[2]

data = pd.read_csv(data_csv)
static_maes = list()
dynamic_maes = list()
for _, row in data.iterrows():
    static_maes.append(row["RibonanzaNetMAE"])
    dynamic_maes.append(row["MAE"])

plt.figure(figsize=(5, 5), dpi=300)
sns.kdeplot(dynamic_maes, fill=True, label="DynamicFold")
sns.kdeplot(static_maes, fill=True, label="RibonanzaNet")
plt.xlabel("MAE per Sequence")
plt.ylabel("Density")
plt.legend()
plt.title("Distribution of MAE on Test Set")
plt.savefig(output_plot)
