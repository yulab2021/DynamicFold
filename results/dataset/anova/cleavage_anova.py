import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pingouin as pg
import sys
import orjson

dataset_csv = sys.argv[1]
levels = int(sys.argv[2])
eps = float(sys.argv[3])
output_plot = sys.argv[4]
output_table = sys.argv[5]

dataset = pd.read_csv(dataset_csv)
reactivity = list()
end_rate = list()
for _, row in dataset.iterrows():
    reactivity.extend(orjson.loads(row["RT"]))
    end_rate.extend(orjson.loads(row["ER"]))

reactivity = np.array(reactivity)
end_rate = np.array(end_rate)
step_size = len(reactivity) / levels
indices = np.argsort(reactivity)
groups = dict()

for i in range(levels):
    group_indices = indices[int(np.ceil(i * step_size)):int(np.ceil((i + 1) * step_size))]
    group_label = f"[{reactivity[group_indices[0]]:.2f}, {reactivity[group_indices[-1]]:.2f}]"
    groups[group_label] = end_rate[group_indices]

groups_scaled = {key: np.log10(val + eps) for key, val in groups.items()}
plt.figure(figsize=(5, 5), dpi=300)
plt.violinplot(list(groups_scaled.values()), positions=list(range(len(groups))), widths=0.8, showmeans=True)
plt.xticks(range(len(groups)), list(groups.keys()), rotation=10, ha="right")
plt.xlabel("Reactivity Score")
plt.ylabel(f"$\\log_{{10}}$(Cleavage Rate + {eps})")
plt.title("Distribution of Cleavage Rate Binned by Reactivity Score")
plt.savefig(output_plot)

groups_melted = {"Group": list(), "Value": list()}
for label, values in groups.items():
    groups_melted["Group"].extend([label] * len(values))
    groups_melted["Value"].extend(values)
df = pd.DataFrame(groups_melted)
aov_table = pg.anova(data=df, dv='Value', between='Group', detailed=True)
print(aov_table)
aov_table.to_csv(output_table, index=False)
