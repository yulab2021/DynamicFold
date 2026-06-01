import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

saliency_csv = sys.argv[1]
aggregation_method = sys.argv[2]
output_plot = sys.argv[3]

normalized_saliencies = pd.read_csv(saliency_csv, index_col=0)
labels = normalized_saliencies.columns.tolist()
normalized_saliencies = np.log10(normalized_saliencies.to_numpy()).T.tolist()

plt.figure(figsize=(8, 5), dpi=300)
plt.violinplot(normalized_saliencies, positions=list(range(len(labels))), widths=0.8, showmeans=True)
plt.xticks(range(len(labels)), labels, rotation=10, ha="right")
plt.ylabel(f"$\\log_{{10}}$(Normalized {aggregation_method} Saliency)")
plt.title("Distribution of Normalized Feature Saliency per Sequence")
plt.tight_layout()
plt.savefig(output_plot)
