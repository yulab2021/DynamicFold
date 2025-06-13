# expression.py

import sqlite3
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

dataset_db = sys.argv[1]
table_name = sys.argv[2]
expression_csv = sys.argv[3]
tsne_csv = sys.argv[4]

def winsorize_scale(values, lower_percentile, upper_percentile):
    lower_threshold = np.percentile(values, lower_percentile)
    upper_threshold = np.percentile(values, upper_percentile)

    if lower_threshold == upper_threshold:
        lower_threshold = np.min(values)
        upper_threshold = np.max(values)
        winsorized_values = values
    else:
        winsorized_values = np.clip(values, lower_threshold, upper_threshold)

    if lower_threshold == upper_threshold != 0:
        scaled_values = winsorized_values / upper_threshold
    elif lower_threshold == upper_threshold == 0:
        scaled_values = winsorized_values
    else:
        scaled_values = (winsorized_values - lower_threshold) / (upper_threshold - lower_threshold)

    return scaled_values

def plot_hist(values, title, xlabel, ylabel, output_png):
    plt.figure(figsize=(10, 5), dpi=300)
    plt.hist(values, bins=100, edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output_png)
    plt.close()

def plot_scatter(df, title, xlabel, ylabel, output_png):
    x = df[xlabel]
    y = df[ylabel]
    plt.figure(figsize=(10, 10), dpi=300)
    plt.scatter(x, y, s=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output_png)
    plt.close()

# Load data
dataset = sqlite3.connect(dataset_db)
cursor = dataset.cursor()
cursor.execute(f"SELECT DISTINCT Sample FROM {table_name}")
samples = [row[0] for row in cursor.fetchall()]
cursor.execute("SELECT DISTINCT RefName FROM ref")
ref_names = [row[0] for row in cursor.fetchall()]
cursor.execute(f"SELECT Sample, RefName, MeanDepth FROM {table_name}")
mean_depths = cursor.fetchall()
dataset.close()

# Process data
expression_data = {sample: {ref_name: 0 for ref_name in ref_names} for sample in samples}
for sample, ref_name, mean_depth in mean_depths:
    expression_data[sample][ref_name] = mean_depth
expression = pd.DataFrame(expression_data)
expression.index.name = "RefName"

# Plot data
for sample in samples:
    plot_hist(np.log10(expression.loc[expression[sample] > 0, sample] + 1), f"{sample} Expression Distribution", r"$\log_{10}$(MeanDepth + 1)", "Frequency", f"{sample}.png")

# Normalize data
for sample in samples:
    expression[sample] = winsorize_scale(np.log10(expression[sample] + 1), 1, 99)

# Save data
expression.to_csv(expression_csv)

# t-SNE
tsne = TSNE(n_components=2)
expression_tsne = tsne.fit_transform(expression.to_numpy())
tsne_df = pd.DataFrame(expression_tsne, index=expression.index, columns=['t-SNE1', 't-SNE2'])
tsne_df.index.name = "RefName"
tsne_df.to_csv(tsne_csv)
plot_scatter(tsne_df, "t-SNE of Expression Data", "t-SNE1", "t-SNE2", "t-SNE.png")
