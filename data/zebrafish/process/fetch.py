# preprocess.py

import multiprocessing
import subprocess
from tqdm import tqdm
import pandas as pd
import sys
import ast

label_csv = sys.argv[1]
num_cores = int(sys.argv[2])

logs = open("logs.txt", "a")
srr_list = pd.read_csv(label_csv)["SRR"].tolist()
srr_list = [srr for item in srr_list for srr in ast.literal_eval(item)]

mv_script = """
find -L raw -mindepth 2 -type f -name "*.sra" -print0 | 
while IFS= read -r -d '' file; do
    mv "$file" raw
done
find -L raw -mindepth 1 -type d -exec rm -rf {} +
"""

# def download_sra(srr):
#     subprocess.run(["prefetch", "--max-size", "100G", srr, "--output-directory", "raw"], stdout=logs, stderr=logs)

# subprocess.run(["mkdir", "-p", "raw", "fastq"], stdout=logs, stderr=logs)

# with multiprocessing.Pool(processes=num_cores) as pool, tqdm(total=len(srr_list), desc="Download SRA") as pbar:
#     for _ in pool.imap_unordered(download_sra, srr_list):
#         pbar.update(1)

# subprocess.run(mv_script, shell=True, executable='/bin/bash', stdout=logs, stderr=logs)

for srr in tqdm(srr_list, desc="Decompress SRA"):
    subprocess.run(["fasterq-dump", "raw/" + srr + ".sra", "--split-files", "--outdir", "fastq/", "--threads", str(num_cores)], stdout=logs, stderr=logs)