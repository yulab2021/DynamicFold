# annotations.py

import re
import pandas as pd
import requests
import sys
import multiprocessing as mp
from tqdm import tqdm

input_fasta = sys.argv[1]
annotations_csv = sys.argv[2]
num_cores = int(sys.argv[3])
batch_size = int(sys.argv[4])

def fetch_chunk(chunk):
    response = requests.post(
        "https://rest.ensembl.org/lookup/id",
        json={"ids": chunk},
    )
    if response.status_code == 200:
        return list(response.json().values())
    else:
        raise requests.HTTPError(f'HTTP request failed with status code {response.status_code}: {response.text}')

def get_transcript_info(ids):
    args = [ids[i:(i + batch_size)] for i in range(0, len(ids), batch_size)]
    with mp.Pool(processes=num_cores) as pool:
        all_data = list(tqdm(pool.imap_unordered(fetch_chunk, args), total=len(args), desc="Fetch Data"))
    return [item for sublist in all_data for item in sublist if item is not None]

with open(input_fasta, "r") as f:
    ids = re.findall(r"(?<=transcript:)(\w+)(?=\s|$)", f.read())

annotations_data = get_transcript_info(ids)
annotations = pd.DataFrame(annotations_data)
annotations.to_csv(annotations_csv, index=False)