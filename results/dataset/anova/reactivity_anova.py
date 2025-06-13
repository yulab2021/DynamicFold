import pandas as pd
import numpy as np
import pingouin as pg
import sys
import orjson

dataset_csv = sys.argv[1]
levels = int(sys.argv[2])
output_table = sys.argv[3]

dataset = pd.read_csv(dataset_csv)
data = {"RD": list(), "ER": list(), "MR": list(), "RibonanzaNetPredictions": list(), "RT": list()}
for _, row in dataset.iterrows():
    for key in data:
        data[key].extend(orjson.loads(row[key]))

for key, value in data.items():
    if key != "RT":
        step_size = len(data[key]) / levels
        indices = np.argsort(data[key])
        for i in range(levels):
            group_indices = indices[int(np.ceil(i * step_size)):int(np.ceil((i + 1) * step_size))]
            group_label = f"[{data[key][group_indices[0]]:.2f}, {data[key][group_indices[-1]]:.2f}]"
            for j in group_indices:
                data[key][j] = group_label

df = pd.DataFrame(data)
aov_table = pg.anova(data=df, dv='RT', between=['RD', 'ER', 'MR', 'RibonanzaNetPredictions'], detailed=True)
print(aov_table)
aov_table.to_csv(output_table, index=False)
