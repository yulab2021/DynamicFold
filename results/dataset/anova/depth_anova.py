import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pingouin as pg
import sys
import orjson

dataset_csv = sys.argv[1]
levels = int(sys.argv[2])
output_plot = sys.argv[3]
output_table = sys.argv[4]

dataset = pd.read_csv(dataset_csv)
reactivity = list()
read_depth = list()
for _, row in dataset.iterrows():
    reactivity.extend(orjson.loads(row["RT"]))
    read_depth.extend(orjson.loads(row["RD"]))

reactivity = np.array(reactivity)
read_depth = np.array(read_depth)
step_size = len(reactivity) / levels
indices = np.argsort(reactivity)
groups = dict()

for i in range(levels):
    group_indices = indices[int(np.ceil(i * step_size)):int(np.ceil((i + 1) * step_size))]
    group_label = f"[{reactivity[group_indices[0]]:.2f}, {reactivity[group_indices[-1]]:.2f}]"
    groups[group_label] = read_depth[group_indices]

plt.figure(figsize=(5, 5), dpi=300)
plt.violinplot(list(groups.values()), positions=list(range(len(groups))), widths=0.8, showmeans=True)
plt.xticks(range(len(groups)), list(groups.keys()), rotation=10, ha="right")
plt.xlabel("Reactivity Score")
plt.ylabel("Winsorize-Scaled Read Depth")
plt.title("Distribution of Read Depth Binned by Reactivity Score")
plt.savefig(output_plot)

groups_melted = {"Group": list(), "Value": list()}
for label, values in groups.items():
    groups_melted["Group"].extend([label] * len(values))
    groups_melted["Value"].extend(values)
df = pd.DataFrame(groups_melted)
aov_table = pg.anova(data=df, dv='Value', between='Group', detailed=True)
print(aov_table)
aov_table.to_csv(output_table, index=False)
