# transcriptome.py

import re
import pandas as pd
import requests
from Bio import SeqIO
import sys
from multiprocessing import Pool
from tqdm import tqdm

input_file = sys.argv[1]
output_file = sys.argv[2]
fetch_log = sys.argv[3]

def fetch_chunk(chunk):
    """Fetches transcript info for a chunk of IDs."""
    response = requests.post(
        "https://rest.ensembl.org/lookup/id",
        json={"ids": chunk},
    )
    if response.status_code == 200:
        return list(response.json().values())  # Convert to list
    else:
        return []

def get_transcript_info(ids, chunk_size=100):
    """Fetches transcript info from Ensembl API in batches using multiprocessing."""
    with Pool() as pool:
        # Process chunks in parallel with progress bar
        all_data = list(tqdm(pool.imap_unordered(fetch_chunk, [ids[i:i+chunk_size] for i in range(0, len(ids), chunk_size)]), total=len(ids)//chunk_size+1))
    
    # Flatten the results
    return [item for sublist in all_data for item in sublist]

def extract_longest_transcript_per_gene(df):
    """Identifies longest transcript per gene."""
    return df.loc[df.groupby("Parent")["length"].idxmax()]

# Read transcripts and extract IDs
with open(input_file, "r") as f:
    ids = re.findall(r"(?<=transcript:)(\w+)(?=\s|$)", f.read())

# Get transcript information
transcript_data = get_transcript_info(ids)

# Create DataFrame and filter for longest transcripts
df = pd.DataFrame(transcript_data)
df.to_csv(fetch_log, index=False)
longest_transcripts = extract_longest_transcript_per_gene(df)

# Write filtered FASTA
with open(output_file, "w") as out_fa:
    for record in SeqIO.parse(input_file, "fasta"):
        transcript_id = record.id.split(":")[1]  
        if transcript_id in longest_transcripts["id"].values:
            description = longest_transcripts.loc[longest_transcripts['id'] == transcript_id, 'display_name'].fillna('').iloc[0]
            record.id = f"{transcript_id}"
            record.name = f"{description}"
            record.description = f"{description}"
            SeqIO.write(record, out_fa, "fasta")
