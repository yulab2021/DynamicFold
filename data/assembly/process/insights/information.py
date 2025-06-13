# informtiation.py

import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import sqlite3
import sys
import orjson
import ast
from tqdm import tqdm

database_db = sys.argv[1]
table_name = sys.argv[2]
key_col = sys.argv[3]
features = ast.literal_eval(sys.argv[4])
clause = sys.argv[5]
output_dir = sys.argv[6]

conn = sqlite3.connect(database_db)
cursor = conn.cursor()
cursor.execute(f"SELECT {', '.join(features)} FROM {table_name} {clause}")
rows = cursor.fetchall()

data = {feature: list() for feature in features}

for row in tqdm(rows, desc="Load Data"):
    for feature, value in zip(features, row):
        data[feature].extend(orjson.loads(value))

data = pd.DataFrame(data)
column_names = list(data.columns)
data = data.to_numpy()
n_features = data.shape[1]
mi_matrix = list()

for i in tqdm(range(n_features), desc="Compute MI"):
    mi_matrix.append(mutual_info_regression(data, data[:, i]))

mi_matrix = pd.DataFrame(mi_matrix, columns=column_names, index=column_names)
mi_matrix.to_csv(f'{output_dir}/information.csv', index=True)
