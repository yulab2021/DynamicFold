import ViennaRNA
import pandas as pd
from tqdm import tqdm
import orjson
import sys

data_csv = sys.argv[1]

def predict_structure(row, m=1.9, b=-0.7):
    fc = ViennaRNA.fold_compound(row["SQ"])
    plain, _ = fc.mfe()
    
    fc.sc_add_SHAPE_deigan(orjson.loads(row["RT"]), m, b)
    experimental, _ = fc.mfe()
    fc.sc_remove()

    fc.sc_add_SHAPE_deigan(orjson.loads(row["Predictions"]), m, b)
    dynamic_fold, _ = fc.mfe()
    fc.sc_remove()

    fc.sc_add_SHAPE_deigan(orjson.loads(row["RibonanzaNetPredictions"]), m, b)
    ribonanza_net, _ = fc.mfe()
    fc.sc_remove()

    return plain, experimental, dynamic_fold, ribonanza_net

data = pd.read_csv(data_csv)
data["PlainStructure"] = [""] * len(data.index)
data["ExperimentalStructure"] = [""] * len(data.index)
data["DynamicFoldStructure"] = [""] * len(data.index)
data["RibonanzaNetStructure"] = [""] * len(data.index)
for index, row in tqdm(data.iterrows(), total=len(data.index), desc="Predict Structure"):
    plain, experimental, dynamic_fold, ribonanza_net = predict_structure(row)
    data.loc[index, "PlainStructure"] = plain
    data.loc[index, "ExperimentalStructure"] = experimental
    data.loc[index, "DynamicFoldStructure"] = dynamic_fold
    data.loc[index, "RibonanzaNetStructure"] = ribonanza_net

data.to_csv(data_csv, index=False)
