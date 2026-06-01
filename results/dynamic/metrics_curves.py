import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import sys
import pandas as pd
import multiprocessing as mp
import sqlite3
import io

data_csv = sys.argv[1]
bpps_db = sys.argv[2]
output_basename = sys.argv[3]
num_cores = int(sys.argv[4])

def proba_matrix(mat):
    buffer = io.BytesIO(mat)
    mat = np.load(buffer).flatten().tolist()
    return mat

def resample(x, y, res=101):
    xp = np.linspace(min(x), max(x), num=res)
    yp = np.interp(xp, x, y)
    return xp, yp

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

def process(args):
    _, row = args
    cursor.execute("SELECT PlainMatrix, RibonanzaNetMatrix, DynamicFoldMatrix FROM bpps WHERE SeqID = ?", (row["SeqID"],))
    matrices = cursor.fetchall()[0]
    plain_proba = proba_matrix(matrices[0])
    static_proba = proba_matrix(matrices[1])
    dynamic_proba = proba_matrix(matrices[2])
    label = basepair_matrix(row["ExperimentalStructure"])

    plain_fpr, plain_tpr, _ = roc_curve(label, plain_proba)
    static_fpr, static_tpr, _ = roc_curve(label, static_proba)
    dynamic_fpr, dynamic_tpr, _ = roc_curve(label, dynamic_proba)

    plain_pr, plain_re, _ = precision_recall_curve(label, plain_proba)
    static_pr, static_re, _ = precision_recall_curve(label, static_proba)
    dynamic_pr, dynamic_re, _ = precision_recall_curve(label, dynamic_proba)

    return [resample(plain_fpr, plain_tpr), resample(static_fpr, static_tpr), resample(dynamic_fpr, dynamic_tpr)], [resample(plain_pr, plain_re), resample(static_pr, static_re), resample(dynamic_pr, dynamic_re)]


data = pd.read_csv(data_csv)
conn = sqlite3.connect(bpps_db)
cursor = conn.cursor()

roc_data = list() # (seq, pipeline (3), x/y (2), 101)
pr_data = list()

with mp.Pool(num_cores) as pool:
    for roc, pr in tqdm(pool.imap_unordered(process, data.iterrows()), total=len(data.index), desc="Load Data"):
        roc_data.append(roc)
        pr_data.append(pr)

roc_data = np.array(roc_data)
roc_mean = roc_data.mean(axis=0)
roc_std = roc_data.std(axis=0)

plt.figure(figsize=(5, 5), dpi=300)
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, (index, name) in enumerate({2: "DynamicFold", 1: "RibonanzaNet", 0: "RNAFold"}.items()):
    line_color = color_cycle[i]
    plt.fill_between(roc_mean[index][0], roc_mean[index][1] - roc_std[index][1], roc_mean[index][1] + roc_std[index][1], color=line_color, alpha=0.3, label=f"{name} (Mean $\\pm$ SD)")
    plt.plot(roc_mean[index][0], roc_mean[index][1], color=line_color, label=f'{name} (Mean AUC = {auc(roc_mean[index][0], roc_mean[index][1]):.2f})')

plt.plot([0, 1], [0, 1], color='black', lw=1, alpha=0.8, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.title('ROC Curve of Secondary Structure Prediction Workflows')
plt.legend()
plt.savefig(f"{output_basename}_ROC.png")


pr_data = np.array(pr_data)
pr_mean = pr_data.mean(axis=0)
pr_std = pr_data.std(axis=0)

plt.figure(figsize=(5, 5), dpi=300)
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, (index, name) in enumerate({2: "DynamicFold", 1: "RibonanzaNet", 0: "RNAFold"}.items()):
    line_color = color_cycle[i]
    plt.fill_between(pr_mean[index][0], pr_mean[index][1] - pr_std[index][1], pr_mean[index][1] + pr_std[index][1], color=line_color, alpha=0.3, label=f"{name} (Mean $\\pm$ SD)")
    plt.plot(pr_mean[index][0], pr_mean[index][1], color=line_color, label=f'{name} (Mean AUC = {auc(pr_mean[index][0], pr_mean[index][1]):.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.title('PR Curve of Secondary Structure Prediction Workflows')
plt.legend()
plt.savefig(f"{output_basename}_PR.png")
