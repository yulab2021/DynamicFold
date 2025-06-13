import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

data_csv = sys.argv[1]
output_plot = sys.argv[2]

data = pd.read_csv(data_csv)
mean_mismatch = list()
MAEs = list()
for _, row in data.iterrows():
    mean_mismatch.append(row["MeanMismatch"])
    MAEs.append(row["MAE"])

plt.figure(figsize=(5, 5), dpi=300)
plt.scatter(np.log10(mean_mismatch), MAEs, s=1, marker="o")
plt.ylim((0, 0.4))
plt.xlabel("$\\log_{10}$(Mean Mismatch Rate)")
plt.ylabel("DynamicFold MAE")
plt.title("Mean Mismatch Rate vs. DynamicFold MAE per Sequence")
plt.savefig(output_plot)
