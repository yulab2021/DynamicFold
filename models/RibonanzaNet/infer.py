import sys
import orjson

arg_json = sys.argv[1]
args = orjson.loads(open(arg_json, "r").read())

from torch.utils.data import Dataset
import sqlite3
import torch
import numpy as np
import yaml
from Network import *
from tqdm import tqdm

class RNADatasetRN(Dataset):
    def __init__(self, dataset_db, table_name, clause):
        self.conn = sqlite3.connect(dataset_db)
        self.cursor = self.conn.cursor()
        self.table_name = table_name
        self.tokens={nt: i for i, nt in enumerate('ACGU')}

        self.cursor.execute(f"PRAGMA table_info({self.table_name})")
        col_names = [col[1] for col in self.cursor.fetchall()]
        if f"RibonanzaNetPredictions" not in col_names:
            self.cursor.execute(f"ALTER TABLE {self.table_name} ADD COLUMN RibonanzaNetPredictions TEXT")
        if f"RibonanzaNetMAE" not in col_names:
            self.cursor.execute(f"ALTER TABLE {self.table_name} ADD COLUMN RibonanzaNetMAE REAL")
        self.conn.commit()

        self.cursor.execute(f"SELECT SeqID FROM {self.table_name} WHERE RibonanzaNetPredictions IS NULL OR RibonanzaNetPredictions = ''")
        null_keys = [row[0] for row in self.cursor.fetchall()]
        self.cursor.execute(f"SELECT SeqID FROM {self.table_name} {clause}")
        filtered_keys = [row[0] for row in self.cursor.fetchall()]
        self.keys = list(set.intersection(set(null_keys), set(filtered_keys)))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        self.cursor.execute(f"SELECT Sequence, RT FROM {self.table_name} WHERE SeqID = ?;", (self.keys[index],))
        sequence, reactivity = self.cursor.fetchone()

        sequence = [self.tokens[nt] for nt in sequence]
        sequence = torch.tensor(np.array(sequence))
        reactivity = orjson.loads(reactivity)

        return sequence, reactivity
    
    def write(self, index, prediction, MAE):
        key = self.keys[index]
        prediction = orjson.dumps(prediction)
        self.cursor.execute(f"UPDATE {self.table_name} SET RibonanzaNetPredictions = ?, RibonanzaNetMAE = ? WHERE SeqID = ?", (prediction, MAE, key))
        self.conn.commit()

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries = entries

    def print(self):
        print(self.entries)

    @classmethod
    def from_yaml(cls, file_path):
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return cls(**config)
    
def mean_absolute_error(prediction, truth):
    MAE = 0
    for p, t in zip(prediction, truth):
        MAE += abs(p - t)
    MAE = MAE / len(prediction)
    return MAE
device = torch.device(args["device"])
dataset = RNADatasetRN(args["dataset_db"], args["table_name"], args["clause"])
model = RibonanzaNet(Config.from_yaml(args["config"]))
model.load_state_dict(torch.load(args["weights_pt"], weights_only=True, map_location=torch.device('cpu')))
model = model.to(device)
model.eval()

with torch.no_grad():
    for index in tqdm(range(len(dataset)), desc="Predict Reactivity"):
        try:
            sequence, reactivity = dataset[index]
            sequence = sequence.unsqueeze(0).to(device)
            prediction = model(sequence, torch.ones_like(sequence)).squeeze().cpu().numpy()
        except KeyError:
            continue
        prediction = prediction[:,0].tolist()
        MAE = mean_absolute_error(prediction, reactivity)
        dataset.write(index, prediction, MAE)
