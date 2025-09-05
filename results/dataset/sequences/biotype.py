import sys
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text

dataset_csv = sys.argv[1]
annotations_csv = sys.argv[2]
label_top = int(sys.argv[3])
autopct_limit = float(sys.argv[4])
output_plot = sys.argv[5]

dataset = pd.read_csv(dataset_csv)
annotations = pd.read_csv(annotations_csv)
biotypes = pd.merge(dataset, annotations[["id", "biotype"]], left_on="RefName", right_on="id", how='left')["biotype"]
biotype_counts = biotypes.value_counts().to_dict()

sizes = list(biotype_counts.values())
labels = list(biotype_counts.keys())
pie_labels = [l.replace("_", " ") if i < label_top else None for i, l in enumerate(labels)]

def autopct_func(value):
    if value >= autopct_limit:
        return f"{value:.1f}%"
    else:
        return None

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

# Create the pie chart
patches, label_texts, autotexts = ax.pie(
    sizes,
    labels=pie_labels,
    labeldistance=1.0,
    autopct=autopct_func,
    pctdistance=0.7,
    startangle=90,
    counterclock=False
)

# Use adjustText to repel the text objects
adjust_text(label_texts[:label_top], ax=ax, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

ax.set_title("Distribution of Sequence Biotypes")
ax.axis('equal')
plt.tight_layout()
plt.savefig(output_plot)
