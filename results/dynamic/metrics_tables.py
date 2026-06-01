import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pandas as pd

data_csv = sys.argv[1]
num_bins = int(sys.argv[2])
output_basename = sys.argv[3]
num_cores = int(sys.argv[4])

def basepair_matrix(dot_bracket):
    seq_len = len(dot_bracket)
    stack = list()
    mat = np.zeros((seq_len, seq_len))
    for i, c in enumerate(dot_bracket):
        if c == '(':
            stack.append(i)
        elif c == ')':
            j = stack.pop(-1)
            mat[i, j] = 1
            mat[j, i] = 1
    return mat.flatten().tolist()

def process(arg):
    _, row = arg
    label = basepair_matrix(row["ExperimentalStructure"])
    plain = basepair_matrix(row["PlainStructure"])
    static = basepair_matrix(row["RibonanzaNetStructure"])
    dynamic = basepair_matrix(row["DynamicFoldStructure"])

    f1_plain = f1_score(label, plain)
    f1_static = f1_score(label, static)
    f1_dynamic = f1_score(label, dynamic)

    accuracy_plain = accuracy_score(label, plain)
    accuracy_static = accuracy_score(label, static)
    accuracy_dynamic = accuracy_score(label, dynamic)

    precision_plain = precision_score(label, plain)
    precision_static = precision_score(label, static)
    precision_dynamic = precision_score(label, dynamic)

    recall_plain = recall_score(label, plain)
    recall_static = recall_score(label, static)
    recall_dynamic = recall_score(label, dynamic)

    confusion_plain = confusion_matrix(label, plain, normalize="true").flatten().tolist()
    confusion_static = confusion_matrix(label, static, normalize="true").flatten().tolist()
    confusion_dynamic = confusion_matrix(label, dynamic, normalize="true").flatten().tolist()

    return [
        [f1_plain, accuracy_plain, precision_plain, recall_plain],
        [f1_static, accuracy_static, precision_static, recall_static],
        [f1_dynamic, accuracy_dynamic, precision_dynamic, recall_dynamic],
    ], [confusion_plain, confusion_static, confusion_dynamic]


data = pd.read_csv(data_csv)
metrics_data = list() # (seq, pipeline, metrics)
confusion_data = list()

with mp.Pool(num_cores) as pool:
    for metrics, confusion in tqdm(pool.imap_unordered(process, data.iterrows()), total=len(data.index), desc="Load Data"):
        metrics_data.append(metrics)
        confusion_data.append(confusion)

metrics_data = np.array(metrics_data).transpose((2, 1, 0))[:,::-1,:]
metrics_df = pd.DataFrame({
    "Pipeline": np.tile(
        np.repeat(["DynamicFold", "RibonanzaNet", "RNAFold"], metrics_data.shape[2]),
        metrics_data.shape[0]
    ), 
    "Metrics": np.repeat(
        ["F1 Score", "Accuracy", "Precision", "Recall"],
        metrics_data.shape[1] * metrics_data.shape[2]
    ),
    "Score": metrics_data.ravel()
})

for metric in metrics_df["Metrics"].unique():
    plt.figure(figsize=(5, 5), dpi=300)
    sns.histplot(metrics_df.loc[metrics_df["Metrics"] == metric], x="Score", hue="Pipeline", bins=num_bins, element="step")
    plt.title(f"Distribution of {metric}")
    plt.tight_layout()
    plt.savefig(f"{output_basename}_{metric.replace(" ", "_").lower()}.png")


confusion_data = np.array(confusion_data).transpose((2, 1, 0))[:,::-1,:]
confusion_df = pd.DataFrame({
    "Pipeline": np.tile(
        np.repeat(["DynamicFold", "RibonanzaNet", "RNAFold"], confusion_data.shape[2]),
        confusion_data.shape[0]
    ), 
    "Confusion": np.repeat(
        ["TNR", "FPR", "FNR", "TPR"],
        confusion_data.shape[1] * confusion_data.shape[2]
    ),
    "Rate": confusion_data.ravel()
})

for metric in confusion_df["Confusion"].unique():
    plt.figure(figsize=(5, 5), dpi=300)
    sns.histplot(confusion_df.loc[confusion_df["Confusion"] == metric], x="Rate", hue="Pipeline", bins=num_bins, element="step")
    plt.title(f"Distribution of {metric}")
    plt.tight_layout()
    plt.savefig(f"{output_basename}_{metric.replace(" ", "_").lower()}.png")
