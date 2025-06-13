# query.py

import sqlite3
import pandas as pd
import sys

database_db = sys.argv[1]
output_csv = sys.argv[2]
query = sys.argv[3]

conn = sqlite3.connect(database_db)
cursor = conn.cursor()
cursor.execute(query)
rows = cursor.fetchall()
col_names = [description[0] for description in cursor.description]
conn.close()

df = dict()
for col_name in col_names:
    df[col_name] = list()
for row in rows:
    for col_name, value in zip(col_names, row):
        if type(value) == bytes:
            value = value.decode()
        df[col_name].append(value)

df = pd.DataFrame(df)
df.to_csv(output_csv, index=False)