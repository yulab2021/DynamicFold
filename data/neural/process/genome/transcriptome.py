# transcriptome.py

import re
import pandas as pd
from Bio import SeqIO
import sys
import subprocess as sp
from tqdm import tqdm

input_fasta = sys.argv[1]
output_fasta = sys.argv[2]
annotations_csv = sys.argv[3]

def count_transcripts(input_fasta):
    return int(sp.check_output(f"grep -c '^>' {input_fasta}", shell=True).decode("utf-8"))

annotations = pd.read_csv(annotations_csv)
longest_transcripts = annotations.loc[annotations.groupby("Parent")["length"].idxmax()]
total_transcripts = count_transcripts(input_fasta)

with open(output_fasta, "w") as output:
    for record in tqdm(SeqIO.parse(input_fasta, "fasta"), total=total_transcripts, desc="Filter Transcripts"):
        transcript_id = re.findall(r"(?<=transcript:)(\w+)(?=\s|$)", record.id)[0]
        if transcript_id in longest_transcripts["id"].values:
            description = longest_transcripts.loc[longest_transcripts['id'] == transcript_id, 'display_name'].fillna('').iloc[0]
            record.id = transcript_id
            record.description = description
            SeqIO.write(record, output, "fasta")
