import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

dataset_csv = sys.argv[1]
output_plot = sys.argv[2]

dataset = pd.read_csv(dataset_csv)
depth = np.log10(dataset["MeanDepth"].to_list())
density = np.log10(dataset["MeanDensity"].to_list())

plt.figure(figsize=(5, 5), dpi=300)
sns.kdeplot(depth, fill=True, label="RNA-Seq")
sns.kdeplot(density, fill=True, label="icSHAPE DMSO")
plt.xlabel("$\\log_{10}$(Mean Read Depth)")
plt.ylabel("Density")
plt.legend()
plt.title("Distribution of Mean Read Depth per Sequence")
plt.savefig(output_plot)
