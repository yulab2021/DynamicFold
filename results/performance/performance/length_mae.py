import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

data_csv = sys.argv[1]
output_plot = sys.argv[2]

def kernel_regression(x, y, bw):
    from statsmodels.nonparametric.kernel_regression import KernelReg
    kr = KernelReg(y, x, 'c', bw=(bw,))
    x_vals = np.arange(min(x), max(x), (max(x) - min(x)) / 1000)
    y_pred, _ = kr.fit(x_vals)
    plt.plot(x_vals, y_pred, color='black', linestyle='--', alpha=0.8, linewidth=1)

data = pd.read_csv(data_csv)
lengths = list()
MAEs = list()
for _, row in data.iterrows():
    lengths.append(row["ValidLength"])
    MAEs.append(row["Score"])

lengths = np.log10(lengths)
plt.figure(figsize=(5, 5), dpi=300)
plt.scatter(lengths, MAEs, s=1, marker="o")
kernel_regression(lengths, MAEs, 2)
plt.xlabel(r"$\log_{10}$(Sequence Valid Length)")
plt.ylabel("DynamicFold MAE")
plt.ylim((0, 0.4))
plt.title("Sequence Length vs. DynamicFold MAE")
plt.tight_layout()
plt.savefig(output_plot)
