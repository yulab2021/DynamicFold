import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

dataset_csv = sys.argv[1]
num_bins = int(sys.argv[2])
output_plot = sys.argv[3]

dataset = pd.read_csv(dataset_csv)

plt.figure(figsize=(5, 5), dpi=300)
sns.histplot(np.log10(dataset["MeanMismatch"]), bins=num_bins, alpha=1.0, edgecolor="white")
plt.xlabel("$\\log_{10}$(Mean Mismatch Rate)")
plt.ylabel("Frequency")
plt.title("Distribution of Mean Mismatch Rate per Sequence")
plt.tight_layout()
plt.savefig(output_plot)
