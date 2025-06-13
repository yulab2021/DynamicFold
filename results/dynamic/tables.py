import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import orjson
import sys
import pandas as pd

data_csv = sys.argv[1]
output_table = sys.argv[2]

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

data = pd.read_csv(data_csv)
plain_classification = list()
static_classification = list()
dynamic_classification = list()
labels = list()
for _, row in tqdm(data.iterrows(), total=len(data.index), desc="Load Data"):
    labels.extend(basepair_matrix(row["ExperimentalStructure"]))
    plain_classification.extend(basepair_matrix(row["PlainStructure"]))
    static_classification.extend(basepair_matrix(row["RibonanzaNetStructure"]))
    dynamic_classification.extend(basepair_matrix(row["DynamicFoldStructure"]))

table = pd.DataFrame({
    "Confusion Matrix": ["", "", ""],
    "F1 Score": [0.0, 0.0, 0.0],
    "Accuracy": [0.0, 0.0, 0.0],
    "Precision": [0.0, 0.0, 0.0],
    "Recall": [0.0, 0.0, 0.0]
}, index=["Control", "RibonanzaNet", "DynamicFold"])

for index, y_pred in zip(["Control", "RibonanzaNet", "DynamicFold"], [plain_classification, static_classification, dynamic_classification]):
    table.loc[index, "Confusion Matrix"] = orjson.dumps(confusion_matrix(labels, y_pred).tolist()).decode()
    table.loc[index, "F1 Score"] = f1_score(labels, y_pred)
    table.loc[index, "Accuracy"] = accuracy_score(labels, y_pred)
    table.loc[index, "Precision"] = precision_score(labels, y_pred)
    table.loc[index, "Recall"] = recall_score(labels, y_pred)

table.to_csv(output_table)
