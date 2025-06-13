# references.py

import sys
import pandas as pd
import numpy as np
import sqlite3

annotations_csv = sys.argv[1]
dataset_db = sys.argv[2]

annotations = pd.read_csv(annotations_csv)
annotations = annotations.loc[annotations.groupby("Parent")["length"].idxmax()].sort_values(by="id")
dataset = sqlite3.connect(dataset_db)
cursor = dataset.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS ref (RefName TEXT PRIMARY KEY, DisplayName TEXT, Biotype TEXT, Start INT, End INT, Length INT, Strand INT, Parent TEXT, SeqRegionName INT, Canonical INT, GENCODEPrimary INT, AssemblyName TEXT, Version INT)")

for _, row in annotations.iterrows():
    display_name = row["display_name"]
    if display_name is np.nan:
        display_name = row["id"]
    entry = (row["id"], display_name, row["biotype"], row["start"], row["end"], row["length"], row["strand"], row["Parent"], row["seq_region_name"], row["is_canonical"], row["gencode_primary"], row["assembly_name"], row["version"])
    cursor.execute(f"INSERT INTO ref VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", entry)

dataset.commit()
dataset.close()