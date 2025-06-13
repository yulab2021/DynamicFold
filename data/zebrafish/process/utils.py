# utils.py

import pandas as pd
import sqlite3
import orjson
import ast
import subprocess
import datetime

class Label:
    def __init__(self, label_csv):
        self.data = pd.read_csv(label_csv, keep_default_na=False, na_values=[])
        self.sanity_check()
        self.data["SRR"] = self.data["SRR"].apply(ast.literal_eval)

    def sanity_check(self):
        for _, row in self.data.iterrows():
            if row["Layout"] != "SINGLE" and row["Layout"] != "PAIRED":
                raise ValueError(f"invalid library layout type: {row["Layout"]}")
            if (row["Experiment"] == "RNA-Seq" and row["Layout"] != "PAIRED") or (row["Experiment"] == "icSHAPE" and row["Layout"] != "SINGLE"):
                raise ValueError(f"incorrect library layout for {row["GSM"]}")
            if (row["Experiment"] == "RNA-Seq" and row["Group"] != "NA") or (row["Experiment"] == "icSHAPE" and (row["Group"] != "DMSO" and row["Group"] != "NAIN3")):
                raise ValueError(f"incorrect experiment group for {row["GSM"]}")
            
    def get_row(self, srr):
        rows = self.data[self.data["SRR"].apply(lambda x: srr in x)]
        if rows.shape[0] > 1:
            raise ValueError(f"{srr} found in multiple rows")
        elif rows.shape[0] == 0:
            raise ValueError(f"{srr} not found")
        row = rows.iloc[0]
        return row
    
    def unique_values(self, property):
        values = self.data[property].unique().tolist()
        return values

    def get_base_name(self, srr):
        row = self.get_row(srr)
        base_name = f"{row["Sample"]}|{row["Experiment"]}|{row["Group"]}|{srr}"
        return base_name
    
    def are_equal(self, srr, map):
        row = self.get_row(srr)
        indicators = list()
        for property, value in map.items():
            indicators.append(row[property] in value)
        return all(indicators)
    
    def get_srr_list(self, map=dict()):
        if len(map) == 0:
            srr_lists = self.data["SRR"].tolist()
            srr_list = [srr for item in srr_lists for srr in item]
        else:
            filtered_data = self.data
            for property, value in map.items():
                filtered_data = filtered_data[filtered_data[property].isin(value)]
            srr_lists = filtered_data["SRR"].tolist()
            srr_list = [srr for item in srr_lists for srr in item]
        return srr_list

class Database:
    def __init__(self, database_db, table_name):
        self.database_db = database_db
        self.table_name = table_name

    def connect(self):
        self.conn = sqlite3.connect(self.database_db)
        self.cursor = self.conn.cursor()
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {self.table_name} (key TEXT PRIMARY KEY, value TEXT)")

    def write(self, key, value):
        serialized_value = orjson.dumps(value)
        self.cursor.execute(f"INSERT INTO {self.table_name} VALUES (?, ?)", (key, serialized_value))

    def read(self, key):
        self.cursor.execute(f"SELECT value FROM {self.table_name} WHERE key = ?", (key,))
        value = self.cursor.fetchone()[0]
        retrieved = orjson.loads(value)
        return retrieved
    
    def list(self):
        self.cursor.execute(f"SELECT key FROM {self.table_name}")
        keys = [row[0] for row in self.cursor.fetchall()]
        return keys
    
    def close(self):
        self.conn.commit()
        self.conn.close()

class Executer:
    def __init__(self, log_file="logs.txt", executable_path="/bin/bash"):
        self.logs = open(log_file, "a")
        self.executable = executable_path

    def run(self, command, message):
        self.log(str(command))
        subprocess.run(command, stdout=self.logs, stderr=self.logs)
        self.log(message)

    def shell(self, script, message):
        self.log(script)
        subprocess.run(script, shell=True, executable=self.executable, stdout=self.logs, stderr=self.logs)
        self.log(message)

    def log(self, message):
        self.logs.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {message}\n")
        self.logs.flush()
