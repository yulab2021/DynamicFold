import pandas as pd
from Bio import pairwise2
import numpy as np
from tqdm import tqdm
import orjson
import json
import sys

data_csv = sys.argv[1]
cache_csv = sys.argv[2]

def indexed_structure(dot_bracket):
    structure = [-1] * len(dot_bracket)
    stack = list()
    for i, c in enumerate(dot_bracket):
        if c == "(":
            stack.append(i)
        elif c == ")":
            j = stack.pop()
            structure[j] = i
            structure[i] = j
    if len(stack) > 0:
        raise ValueError("unmatched bracket")
    return structure

def alignment_score(seq_A, seq_B):
    assert len(seq_A) == len(seq_B)
    score = pairwise2.align.globalms(seq_A, seq_B, match=1, mismatch=-1, open=-1, extend=-1, gap_char=["-"], score_only=True)
    score = score / len(seq_A)
    return score

data = pd.read_csv(data_csv)
data["PC"] = [0.0] * len(data.index)
data["NC"] = [0.0] * len(data.index)
data["CC"] = [0.0] * len(data.index)
data["SC"] = [0.0] * len(data.index)
data["VR"] = [0.0] * len(data.index)
data["VP"] = [0.0] * len(data.index)

# Accuracy
for i, row in tqdm(data.iterrows(), total=len(data.index)):
    data.loc[i, "SC"] = 1 - row["ExperimentalStructure"].count(".") / len(row["ExperimentalStructure"])
    data.loc[i, "NC"] = alignment_score(indexed_structure(row["PlainStructure"]), indexed_structure(row["ExperimentalStructure"]))
    data.loc[i, "PC"] = alignment_score(indexed_structure(row["DynamicFoldStructure"]), indexed_structure(row["ExperimentalStructure"]))
    data.loc[i, "CC"] = alignment_score(indexed_structure(row["RibonanzaNetStructure"]), indexed_structure(row["ExperimentalStructure"]))

# Dynamicity
ref_names = data["RefName"].unique()
start_max = data["Start"].groupby(data["RefName"]).max()
end_min = data["End"].groupby(data["RefName"]).min()
vr = dict()
vp = dict()

for ref_name in tqdm(ref_names):
    rows = data.loc[data["RefName"] == ref_name]
    reactivity = list()
    dynamic = list()
    for _, row in rows.iterrows():
        reactivity.append(indexed_structure("." * (row["Start"] - 1) + row["ExperimentalStructure"] + "." * (1 - row["End"])))
        dynamic.append(indexed_structure("." * (row["Start"] - 1) + row["DynamicFoldStructure"] + "." * (1 - row["End"])))
    if len(reactivity) <= 1:
        vr[ref_name] = 0
        vp[ref_name] = 0
        continue
    vr_scores = list()
    vp_scores = list()
    for i in range(len(reactivity)):
        for j in range(i+1, len(reactivity)):
            vr_scores.append(alignment_score(reactivity[i], reactivity[j]))
            vp_scores.append(alignment_score(dynamic[i], dynamic[j]))
    vr[ref_name] = np.mean(vr_scores)
    vp[ref_name] = np.mean(vp_scores)
data["VR"] = data["RefName"].map(vr)
data["VP"] = data["RefName"].map(vp)
data = data.sort_values("PC", ascending=False)

data.to_csv(cache_csv, index=False)
data = pd.read_csv(cache_csv)

selection = (data["PC"] >= 0.8) & (data["NC"] < data["PC"]) & (data["CC"] < data["PC"]) & (data["SC"] > 0.4) & (data["VR"] < 0.4) & (data["VP"] < 0.4) & (data["VR"] != 0)
print(data.loc[selection, ["SeqID", "MAE", "ValidLength", "Start", "End", "MeanDepth", "MeanDensity", "biotype"]])
print(data.loc[selection, ["SeqID", "PC", "NC", "CC", "SC", "VR", "VP"]])

data.columns = ['SeqID', 'DynamicFold Predictions', 'DynamicFold MAE', 'Dataset', 'Sequence', 'RD', 'ER', 'MR', 'Reactivity', 'IC', 'Sample', 'RefName', 'Start', 'End', 'FullLength', 'ValidLength', 'StripLength', 'MeanDepth', 'MeanEnd', 'MeanDensity', 'MeanMismatch', 'Gap', 'RibonanzaNet Predictions', 'RibonanzaNet MAE', 'id', 'Biotype', 'Plain Structure', 'Experimental Structure', 'DynamicFold Structure', 'RibonanzaNet Structure', 'PC', 'NC', 'CC', 'SC', 'VR', 'VP']

data.set_index("Sample", drop=False, inplace=True, verify_integrity=False)
entry = {"RefName": "ENSDART00000187534"}
entry.update(data.loc[(data["RefName"] == "ENSDART00000187534") & ((data["Sample"] == "2h-wt") | (data["Sample"] == "4h-wt")), ['Biotype', 'Sequence', 'Start', 'End', 'ValidLength', 'Reactivity', 'DynamicFold Predictions', 'RibonanzaNet Predictions', 'DynamicFold MAE', 'RibonanzaNet MAE', 'Experimental Structure', 'DynamicFold Structure', 'RibonanzaNet Structure', 'Plain Structure', 'MeanDensity', 'MeanDepth', 'MeanEnd', 'MeanMismatch', 'PC', 'NC', 'CC', 'SC', 'VR', 'VP']].to_dict())

with open("ENSDART00000187534.json", "w") as file:
    json.dump(entry, file, indent=4)
