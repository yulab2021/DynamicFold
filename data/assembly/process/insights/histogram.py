# histogram.py

import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import sqlite3
import ast
import numpy as np

database_db = sys.argv[1]
table_name = sys.argv[2]
features = ast.literal_eval(sys.argv[3])
clause = sys.argv[4]
num_bins = int(sys.argv[5])
logarithmic = ast.literal_eval(sys.argv[6])
eps = float(sys.argv[7])
output_dir = sys.argv[8]

conn = sqlite3.connect(database_db)
cursor = conn.cursor()

for feature in tqdm(features, desc="Calculate Features"):
    cursor.execute(f"SELECT {feature} FROM {table_name} {clause}")
    if logarithmic:
        data = [np.log10(abs(float(row[0])) + eps) for row in cursor.fetchall()]
    else:
        data = [float(row[0]) for row in cursor.fetchall()]
    plt.figure(figsize=(10, 5), dpi=300)
    plt.hist(data, bins=num_bins, edgecolor="white")
    if logarithmic:
        plt.xlabel(f"$\\log_{{10}}$(|{feature}| + {eps})")
    else:
        plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"{table_name}: {feature}")
    plt.savefig(f"{output_dir}/{table_name}_{feature}.png")

conn.close()
