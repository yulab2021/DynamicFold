import pandas as pd
import matplotlib.pyplot as plt
import sys

data_csv = sys.argv[1]
output_plot = sys.argv[2]

data = pd.read_csv(data_csv)
diff = dict()

for sample in data["Sample"].unique():
    diff[sample] = data.loc[data["Sample"] == sample, "MAE"] - data.loc[data["Sample"] == sample, "RibonanzaNetMAE"]

diff = dict(sorted(diff.items(), key=lambda x: x[0]))

plt.figure(figsize=(10, 5), dpi=300)
plt.violinplot(list(diff.values()), positions=list(range(len(diff))), widths=0.8, showmeans=True)
plt.xticks(range(len(diff)), list(diff.keys()))
plt.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
plt.xlabel("Sample")
plt.ylabel("DynamicFold MAE - RibonanzaNet MAE")
plt.title("Distribution of Difference in Prediction MAE per Sequence")
plt.savefig(output_plot)
