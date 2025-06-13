import matplotlib.pyplot as plt
import pandas as pd
import sys

data_csv = sys.argv[1]
output_plot = sys.argv[2]

data = pd.read_csv(data_csv)
depths = list()
MAEs = list()
for _, row in data.iterrows():
    depths.append(row["MeanDepth"])
    MAEs.append(row["MAE"])

plt.figure(figsize=(5, 5), dpi=300)
plt.scatter(depths, MAEs, s=1, marker="o")
plt.xlabel("Mean Read Depth")
plt.ylabel("DynamicFold MAE")
plt.xlim((32, 1024))
plt.ylim((0, 0.4))
plt.title("Mean Read Depth vs. DynamicFold MAE per Sequence")
plt.savefig(output_plot)
