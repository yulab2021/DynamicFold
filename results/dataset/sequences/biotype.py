import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset_csv = sys.argv[1]
annotations_csv = sys.argv[2]
label_top = int(sys.argv[3])
autopct_limit = float(sys.argv[4])
output_plot = sys.argv[5]

dataset = pd.read_csv(dataset_csv)
annotations = pd.read_csv(annotations_csv)
biotypes = pd.merge(dataset, annotations[["id", "biotype"]], left_on="RefName", right_on="id", how='left')["biotype"].to_list()
biotype_counts = dict()

for biotype in biotypes:
    biotype = biotype.replace("_", " ")
    if biotype not in biotype_counts:
        biotype_counts[biotype] = 0
    biotype_counts[biotype] += 1

sizes = list(biotype_counts.values())
labels = list(biotype_counts.keys())
data = sorted(zip(sizes, labels), key=lambda x: x[0], reverse=True)
sizes = [d[0] for d in data]
labels = [d[1] for d in data]
pie_labels = [label if i < label_top else '' for i, label in enumerate(labels)]

def autopct(value):
    if value >= autopct_limit:
        return f"{value:.1f}%"
    else:
        return ""

plt.figure(figsize=(10, 5), dpi=300)
patches, texts, autotexts = plt.pie(sizes, labels=pie_labels, autopct=autopct, startangle=90, counterclock=False)
for patch, txt in zip(patches, autotexts):
    ang = (patch.theta2 + patch.theta1) / 2
    x = patch.r * 0.8 * np.cos(ang * np.pi / 180)
    y = patch.r * 0.8 * np.sin(ang * np.pi / 180)
    if (patch.theta2 - patch.theta1) < 45:
        txt.set_position((x, y))
plt.title("Distribution of Sequence Biotypes")
plt.savefig(output_plot)
