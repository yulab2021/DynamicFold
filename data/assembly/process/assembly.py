# assembly.py

import sqlite3
import re
import sys

source_db = sys.argv[1]
source_table = sys.argv[2]
target_db = sys.argv[3]
target_table = sys.argv[4]

source_conn = sqlite3.connect(source_db)
source_cursor = source_conn.cursor()
target_conn = sqlite3.connect(target_db)
target_cursor = target_conn.cursor()

source_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{source_table}'")
source_schema = source_cursor.fetchone()
source_schema = re.search(r'\((.*)\)', source_schema[0], re.DOTALL).group(1).strip()

target_cursor.execute(f"CREATE TABLE IF NOT EXISTS {target_table} ({source_schema})")
target_cursor.execute(f"ATTACH DATABASE '{source_db}' AS source")
target_cursor.execute(f"PRAGMA table_info('{target_table}');")
target_pragma = target_cursor.fetchall()
target_columns = [col[1] for col in target_pragma]

insert_columns = ", ".join(target_columns)
select_columns = ", ".join([f"source.{source_table}.{col}" for col in target_columns])

query = f"INSERT INTO {target_table} ({insert_columns}) SELECT {select_columns} FROM source.{source_table}"
print(query)
target_cursor.execute(query)
target_conn.commit()
