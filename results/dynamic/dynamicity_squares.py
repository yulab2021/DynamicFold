import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import orjson

data_csv = sys.argv[1]
num_bins = int(sys.argv[2])
output_plot = sys.argv[3]

data = pd.read_csv(data_csv)
ref_names = data["RefName"].unique()
rms_experimental = list()
rms_static = list()
rms_dynamic = list()

for ref_name in ref_names:
    rows = data.loc[data["RefName"] == ref_name]
    reactivity = rows["RT"].apply(lambda x: orjson.loads(x)).tolist()
    static = rows["RibonanzaNetPredictions"].apply(lambda x: orjson.loads(x)).tolist()
    dynamic = rows["Predictions"].apply(lambda x: orjson.loads(x)).tolist()
    reactivity = np.array(reactivity)
    static = np.array(static)
    dynamic = np.array(dynamic)
    df = reactivity.shape[0] * (reactivity.shape[1] - 1)
    rms_experimental.append(np.sqrt(np.sum((reactivity - reactivity.mean(axis=0))**2) / df))
    rms_static.append(np.sqrt(np.sum((static - static.mean(axis=0))**2) / df))
    rms_dynamic.append(np.sqrt(np.sum((dynamic - dynamic.mean(axis=0))**2) / df))

df = pd.DataFrame({"Model": ["DynamicFold"] * len(rms_dynamic) + ["RibonanzaNet"] * len(rms_static) + ["Experimental"] * len(rms_experimental), "Value": rms_dynamic + rms_static + rms_experimental})
plt.figure(figsize=(5, 5), dpi=300)
plt.yscale("log")
sns.histplot(df, x="Value", hue="Model", bins=num_bins, element="step")
plt.xlabel("Mean RMSD of Reactivity per Transcript")
plt.ylabel("Frequency")
plt.title("Distribution of Secondary Structure Dynamicity")
plt.savefig(output_plot)
