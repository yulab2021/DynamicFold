import sys
import orjson
import sqlite3
import torch
import ufold
from tqdm import tqdm
import io

arg_json = sys.argv[1]
args = orjson.loads(open(arg_json, "r").read())

class Database:
    def __init__(self, database_db, table_name, clause):
        self.conn = sqlite3.connect(database_db)
        self.cursor = self.conn.cursor()
        self.table_name = table_name
        self.buffer = io.BytesIO()

        self.cursor.execute(f"PRAGMA table_info({self.table_name})")
        col_names = [col[1] for col in self.cursor.fetchall()]
        if f"UFoldBPP" not in col_names:
            self.cursor.execute(f"ALTER TABLE {self.table_name} ADD COLUMN UFoldBPP TEXT")
        self.conn.commit()

        self.cursor.execute(f"SELECT SeqID FROM {self.table_name} WHERE UFoldBPP IS NULL")
        null_keys = [row[0] for row in self.cursor.fetchall()]
        self.cursor.execute(f"SELECT SeqID FROM {self.table_name} {clause}")
        filtered_keys = [row[0] for row in self.cursor.fetchall()]
        self.keys = list(set.intersection(set(null_keys), set(filtered_keys)))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        self.cursor.execute(f"SELECT Sequence FROM {self.table_name} WHERE SeqID = ?;", (self.keys[index],))
        sequence, = self.cursor.fetchone()
        return [sequence,]
    
    def write(self, index, utility:torch.Tensor):
        self.buffer.seek(0)
        self.buffer.truncate(0)
        torch.save(utility.to_sparse(), self.buffer)
        utility = self.buffer.getvalue()
        key = self.keys[index]
        self.cursor.execute(f"UPDATE {self.table_name} SET UFoldBPP = ? WHERE SeqID = ?", (utility, key))
        self.conn.commit()

device = torch.device(args["device"])
database = Database(args["database_db"], args["table_name"], args["clause"])
model = ufold.UFold(**args["model_args"])
model = model.to(device)

for index in tqdm(range(len(database)), desc="Caching"):
    try:
        sequence = database[index]
        utility = model(sequence).squeeze(0).cpu()
    except KeyError:
        continue
    if torch.isnan(utility).any():
        continue
    database.write(index, utility)
