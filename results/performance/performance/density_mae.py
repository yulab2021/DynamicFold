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
density = list()
MAEs = list()
for _, row in data.iterrows():
    density.append(row["MeanDensity"])
    MAEs.append(row["Score"])

density = np.log10(density)
plt.figure(figsize=(5, 5), dpi=300)
plt.scatter(density, MAEs, s=1, marker="o", label="Data")
kernel_regression(density, MAEs, 2)
plt.xlabel(r"$\log_{10}$(Mean Background Base Density)")
plt.ylabel("DynamicFold MAE")
plt.xlim((1.75, 4.0))
plt.ylim((0, 0.4))
plt.title("DynamicFold MAE vs. icSHAPE Background per Sequence")
plt.tight_layout()
plt.savefig(output_plot)
