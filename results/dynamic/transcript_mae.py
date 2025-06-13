import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data_csv = sys.argv[1]
output_plot = sys.argv[2]

data = pd.read_csv(data_csv)
dynamic_maes = list()
static_maes = list()
for ref_name in data["RefName"].unique():
    dynamic_maes.append(np.mean(data.loc[data["RefName"] == ref_name, "MAE"]))
    static_maes.append(np.mean(data.loc[data["RefName"] == ref_name, "RibonanzaNetMAE"]))

plt.figure(figsize=(5, 5), dpi=300)
sns.kdeplot(dynamic_maes, fill=True, label="DynamicFold")
sns.kdeplot(static_maes, fill=True, label="RibonanzaNet")
plt.xlabel("Mean MAE of Predicted Reactivity per Reference")
plt.ylabel("Density")
plt.legend()
plt.title("Comparison of Ability to Capture Structure Variability")
plt.savefig(output_plot)
