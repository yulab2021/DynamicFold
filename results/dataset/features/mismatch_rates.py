import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset_csv = sys.argv[1]
num_bins = int(sys.argv[2])
output_plot = sys.argv[3]

dataset = pd.read_csv(dataset_csv)
mean_mismatch = dataset["MeanMismatch"].to_numpy()
mean_mismatch = mean_mismatch[mean_mismatch > 0]

plt.figure(figsize=(5, 5), dpi=300)
plt.hist(np.log10(mean_mismatch), bins=num_bins, edgecolor="white")
plt.xlabel("$\\log_{10}$(Mean Mismatch Rate)")
plt.ylabel("Frequency")
plt.title("Distribution of Mean Mismatch Rate per Sequence")
plt.savefig(output_plot)
