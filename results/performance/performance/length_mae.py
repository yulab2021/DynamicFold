import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg
import sys

data_csv = sys.argv[1]
output_plot = sys.argv[2]

data = pd.read_csv(data_csv)
lengths = list()
MAEs = list()
for _, row in data.iterrows():
    lengths.append(row["ValidLength"])
    MAEs.append(row["MAE"])

kr = KernelReg(MAEs, lengths, 'c', bw=(180,))
lengths_range = np.arange(min(lengths), max(lengths), 0.1)
MAEs_pred, _ = kr.fit(lengths_range)

plt.figure(figsize=(5, 5), dpi=300)
plt.scatter(lengths, MAEs, s=1, marker="o")
plt.plot(lengths_range, MAEs_pred, color='black', linestyle='--', alpha=1)
plt.xlabel("Sequence Valid Length")
plt.ylabel("DynamicFold MAE")
plt.xlim((32, 1024))
plt.ylim((0, 0.4))
plt.title("Sequence Length vs. DynamicFold MAE")
plt.savefig(output_plot)
