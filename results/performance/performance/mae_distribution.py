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
static_maes = list()
dynamic_maes = list()
for _, row in data.iterrows():
    static_maes.append(row["RibonanzaNetMAE"])
    dynamic_maes.append(row["Score"])

plt.figure(figsize=(5, 5), dpi=300)
plt.scatter(static_maes, dynamic_maes, s=1, marker="o")
kernel_regression(static_maes, dynamic_maes, 0.2)
plt.xlabel("RibonanzaNet MAE")
plt.ylabel("DynamicFold MAE")
plt.xlim((0.05, 0.4))
plt.ylim((0.05, 0.4))
plt.title("DynamicFold vs RibonanzaNet MAE per Sequence")
plt.tight_layout()
plt.savefig(output_plot)
