import ViennaRNA
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import orjson
import sys
import pandas as pd

data_csv = sys.argv[1]
bootstrap = int(sys.argv[2])
output_basename = sys.argv[3]

def structure_probability(sequence, icshape=None, m=1.9, b=-0.7):
    fc = ViennaRNA.fold_compound(sequence)
    if icshape is not None:
        fc.sc_add_SHAPE_deigan(icshape, m, b)
    fc.pf()
    probability_matrix = np.array(fc.bpp())[1:,1:].flatten().tolist()
    probability_matrix = np.nan_to_num(probability_matrix, copy=False)
    return probability_matrix

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
indices = np.random.choice(data.index, size=bootstrap, replace=True)
data = data.loc[indices]
plain_classification = list()
plain_probability = list()
static_classification = list()
static_probability = list()
dynamic_classification = list()
dynamic_probability = list()
labels = list()
for _, row in tqdm(data.iterrows(), total=len(data.index), desc="Load Data"):
    labels.extend(basepair_matrix(row["ExperimentalStructure"]))
    plain_classification.extend(basepair_matrix(row["PlainStructure"]))
    static_classification.extend(basepair_matrix(row["RibonanzaNetStructure"]))
    dynamic_classification.extend(basepair_matrix(row["DynamicFoldStructure"]))

    plain_probability.extend(structure_probability(row["SQ"]))
    static_probability.extend(structure_probability(row["SQ"], orjson.loads(row["RibonanzaNetPredictions"])))
    dynamic_probability.extend(structure_probability(row["SQ"], orjson.loads(row["Predictions"])))

plain_fpr, plain_tpr, _ = roc_curve(labels, plain_probability)
plain_auc = auc(plain_fpr, plain_tpr)
static_fpr, static_tpr, _ = roc_curve(labels, static_probability)
static_auc = auc(static_fpr, static_tpr)
dynamic_fpr, dynamic_tpr, _ = roc_curve(labels, dynamic_probability)
dynamic_auc = auc(dynamic_fpr, dynamic_tpr)

plt.figure(figsize=(5, 5), dpi=300)
plt.plot(dynamic_fpr, dynamic_tpr, label=f'DynamicFold (AUC = {dynamic_auc:.2f})')
plt.plot(static_fpr, static_tpr, label=f'RibonanzaNet (AUC = {static_auc:.2f})')
plt.plot(plain_fpr, plain_tpr, label=f'No icSHAPE (AUC = {plain_auc:.2f})')
plt.plot([0, 1], [0, 1], color='black', lw=1, alpha=0.8, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Secondary Structure Prediction Workflows')
plt.legend()
plt.savefig(f"{output_basename}_ROC.png")

plain_pr, plain_re, _ = precision_recall_curve(labels, plain_probability)
plain_auc = auc(plain_re, plain_pr)
static_pr, static_re, _ = precision_recall_curve(labels, static_probability)
static_auc = auc(static_re, static_pr)
dynamic_pr, dynamic_re, _ = precision_recall_curve(labels, dynamic_probability)
dynamic_auc = auc(dynamic_re, dynamic_pr)

plt.figure(figsize=(5, 5), dpi=300)
plt.plot(dynamic_re, dynamic_pr, label=f'DynamicFold (AUC = {dynamic_auc:.2f})')
plt.plot(static_re, static_pr, label=f'RibonanzaNet (AUC = {static_auc:.2f})')
plt.plot(plain_re, plain_pr, label=f'No icSHAPE (AUC = {plain_auc:.2f})')
plt.plot([0, 1], [1, 0], color='black', lw=1, alpha=0.8, linestyle='--')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve of Secondary Structure Prediction Workflows')
plt.legend()
plt.savefig(f"{output_basename}_PR.png")
