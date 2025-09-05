import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

dataset_csv = sys.argv[1]
num_bins = int(sys.argv[2])
output_plot = sys.argv[3]

dataset = pd.read_csv(dataset_csv)
data = {
    "Value": np.log10(pd.concat([dataset["MeanDepth"], dataset["MeanDensity"]])),
    "Experiment": ["RNA-Seq"] * len(dataset) + ["icSHAPE DMSO"] * len(dataset)
}

plt.figure(figsize=(5, 5), dpi=300)
sns.histplot(data, x="Value", hue="Experiment", bins=num_bins, element="step")
plt.xlabel("$\\log_{10}$(Mean Read Depth)")
plt.ylabel("Frequency")
plt.title("Distribution of Mean Read Depth per Sequence")
plt.tight_layout()
plt.savefig(output_plot)
