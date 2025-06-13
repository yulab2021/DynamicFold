import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import orjson

data_csv = sys.argv[1]
output_plot = sys.argv[2]

data = pd.read_csv(data_csv)
ref_names = data["RefName"].unique()
start_max = data["Start"].groupby(data["RefName"]).max()
end_min = data["End"].groupby(data["RefName"]).min()
ms_experimental = list()
ms_static = list()
ms_dynamic = list()

for ref_name in ref_names:
    rows = data.loc[data["RefName"] == ref_name]
    reactivity = list()
    static = list()
    dynamic = list()
    for _, row in rows.iterrows():
        start = start_max[ref_name] - row["Start"]
        end = len(row["SQ"]) + end_min[ref_name] - row["End"]
        reactivity.append(orjson.loads(row["RT"])[start:end])
        static.append(orjson.loads(row["RibonanzaNetPredictions"])[start:end])
        dynamic.append(orjson.loads(row["Predictions"])[start:end])
    reactivity = np.array(reactivity)
    static = np.array(static)
    dynamic = np.array(dynamic)
    df = reactivity.shape[0] * (reactivity.shape[1] - 1)
    ms_experimental.append(np.sum((reactivity - reactivity.mean(axis=0))**2) / df)
    ms_static.append(np.sum((static - static.mean(axis=0))**2) / df)
    ms_dynamic.append(np.sum((dynamic - dynamic.mean(axis=0))**2) / df)

rms_dynamic = np.sqrt(ms_dynamic)
rms_static = np.sqrt(ms_static)
rms_experimental = np.sqrt(ms_experimental)
plt.figure(figsize=(5, 5), dpi=300)
sns.kdeplot(rms_dynamic, fill=True, label="DynamicFold")
sns.kdeplot(rms_static, fill=True, label="RibonanzaNet")
sns.kdeplot(rms_experimental, fill=True, label="Experimental")
plt.xlabel("Mean RMSD of Predicted/Experimental Reactivity per Sequence")
plt.ylabel("Density")
plt.legend()
plt.title("Distribution of Secondary Structure Dynamicity")
plt.savefig(output_plot)
