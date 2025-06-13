import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

dataset_csv = sys.argv[1]
output_plot = sys.argv[2]

dataset = pd.read_csv(dataset_csv)
strip_start = np.log10(np.abs(dataset["Start"]) + 1)
strip_end = np.log10(np.abs(dataset["End"]) + 1)

plt.figure(figsize=(5, 5), dpi=300)
sns.kdeplot(strip_start, fill=True, label="5'")
sns.kdeplot(strip_end, fill=True, label="3'")
plt.xlabel("$\\log_{10}$(Number of Missing Bases + 1)")
plt.ylabel("Density")
plt.legend()
plt.title("Distribution of Missing Data per Sequence")
plt.savefig(output_plot)
