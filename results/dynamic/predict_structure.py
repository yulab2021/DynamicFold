import ViennaRNA
import pandas as pd
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import orjson
import sys
import sqlite3
import io

dataset_csv = sys.argv[1]
structure_csv = sys.argv[2]
bpps_db = sys.argv[3]
num_cores = int(sys.argv[4])

def dump_binary(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()

def predict_structure(args, m=1.9, b=-0.7):
    seq_id, row = args

    fc = ViennaRNA.fold_compound(row["Sequence"])
    plain_struc, plain_mine = fc.mfe()
    fc.pf()
    plain_proba = dump_binary(np.nan_to_num(np.array(fc.bpp())[1:,1:]))
    
    fc = ViennaRNA.fold_compound(row["Sequence"])
    fc.sc_add_SHAPE_deigan(orjson.loads(row["RT"]), m, b)
    experimental_struc, experimental_mine = fc.mfe()
    fc.pf()
    experimental_proba = dump_binary(np.nan_to_num(np.array(fc.bpp())[1:,1:]).tolist())

    fc = ViennaRNA.fold_compound(row["Sequence"])
    fc.sc_add_SHAPE_deigan(orjson.loads(row["Predictions"]), m, b)
    dynamic_struc, dynamic_mine = fc.mfe()
    fc.pf()
    dynamic_proba = dump_binary(np.nan_to_num(np.array(fc.bpp())[1:,1:]).tolist())

    fc = ViennaRNA.fold_compound(row["Sequence"])
    fc.sc_add_SHAPE_deigan(orjson.loads(row["RibonanzaNetPredictions"]), m, b)
    static_struc, static_mine = fc.mfe()
    fc.pf()
    static_proba = dump_binary(np.nan_to_num(np.array(fc.bpp())[1:,1:]).tolist())

    return seq_id, (plain_struc, plain_mine, experimental_struc, experimental_mine, dynamic_struc, dynamic_mine, static_struc, static_mine), (plain_proba, experimental_proba, dynamic_proba, static_proba)

data = pd.read_csv(dataset_csv, index_col=0)
data["PlainStructure"] = [""] * len(data.index)
data["PlainMinE"] = [0.0] * len(data.index)
data["ExperimentalStructure"] = [""] * len(data.index)
data["ExperimentalMinE"] = [0.0] * len(data.index)
data["DynamicFoldStructure"] = [""] * len(data.index)
data["DynamicFoldMinE"] = [0.0] * len(data.index)
data["RibonanzaNetStructure"] = [""] * len(data.index)
data["RibonanzaNetMinE"] = [0.0] * len(data.index)

conn = sqlite3.connect(bpps_db)
cursor = conn.cursor()
cursor.execute("CREATE TABLE bpps (SeqID TEXT PRIMARY KEY, PlainMatrix BLOB, ExperimentalMatrix BLOB, DynamicFoldMatrix BLOB, RibonanzaNetMatrix BLOB)")

with mp.Pool(num_cores) as pool:
    for seq_id, results, matrices in tqdm(pool.imap_unordered(predict_structure, data.iterrows()), total=len(data.index), desc="Predict Structure"):
        try:
            data.loc[seq_id, ["PlainStructure", "PlainMinE", "ExperimentalStructure", "ExperimentalMinE", "DynamicFoldStructure", "DynamicFoldMinE", "RibonanzaNetStructure", "RibonanzaNetMinE"]] = results
            cursor.execute("INSERT INTO bpps (SeqID, PlainMatrix, ExperimentalMatrix, DynamicFoldMatrix, RibonanzaNetMatrix) VALUES (?, ?, ?, ?, ?)", (seq_id, *matrices))
            conn.commit()
        except Exception as e:
            print(e)

data.to_csv(structure_csv)
