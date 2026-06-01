import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import orjson

data_csv = sys.argv[1]
bootstrap_size = int(sys.argv[2])
output_plot = sys.argv[3]

def kernel_regression(x, y, bw):
    from statsmodels.nonparametric.kernel_regression import KernelReg
    kr = KernelReg(y, x, 'c', bw=(bw,))
    x_vals = np.arange(min(x), max(x), (max(x) - min(x)) / 1000)
    y_pred, _ = kr.fit(x_vals)
    plt.plot(x_vals, y_pred, color='black', linestyle='--', alpha=0.8, linewidth=1)

data = pd.read_csv(data_csv)
reactivity = list()
prediction = list()
for _, row in data.iterrows():
    reactivity.extend(orjson.loads(row["RT"]))
    prediction.extend(orjson.loads(row["Predictions"]))

indices = list(range(len(reactivity)))
indices = np.random.choice(indices, size=bootstrap_size, replace=True)
prediction_sample = np.array(prediction)[indices]
reactivity_sample = np.array(reactivity)[indices]

plt.figure(figsize=(5, 5), dpi=300)
plt.scatter(prediction_sample, reactivity_sample, s=0.1, marker="o")
kernel_regression(prediction_sample, reactivity_sample, 0.5)
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.xlabel("DynamicFold Prediction")
plt.ylabel("Reactivity Score")
plt.title("DynamicFold Prediction vs. True Reactivity")
plt.tight_layout()
plt.savefig(output_plot)
