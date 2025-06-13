import sys
import matplotlib.pyplot as plt
import pandas as pd

dataset_csv = sys.argv[1]
num_bins = int(sys.argv[2])
output_plot = sys.argv[3]

dataset = pd.read_csv(dataset_csv)
valid_lengths = dataset["ValidLength"].to_list()

plt.figure(figsize=(5, 5), dpi=300)
plt.hist(valid_lengths, bins=num_bins, edgecolor="white")
plt.xlabel("Valid Length")
plt.ylabel("Frequency")
plt.title("Distribution of Sequence Valid Length")
plt.savefig(output_plot)
