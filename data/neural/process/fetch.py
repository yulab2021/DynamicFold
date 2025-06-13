# fetch.py

import multiprocessing as mp
from tqdm import tqdm
import sys
import utils

label_csv = sys.argv[1]
max_size = sys.argv[2]
num_cores = int(sys.argv[3])
batch_size = int(sys.argv[4])

def download_srr(srr):
    executer.run(["prefetch", "--max-size", max_size, srr, "--output-directory", "raw"], f"{srr} downloaded")

def dump_fastq(srr):
    executer.run(["fasterq-dump", "--force", "--verbose", "--split-files", "--outdir", "fastq/", "--temp", "cache/", "--threads", str(num_cores), f"raw/{srr}/{srr}.sra"], f"{srr} dumped")

def compress_fastq(srr):
    if labels.are_equal(srr, {"Layout": ["PAIRED"]}):
        executer.run(["pigz", "--processes", str(num_cores), f"fastq/{srr}_1.fastq", f"fastq/{srr}_2.fastq"], f"{srr} compressed")
    else:
        executer.run(["pigz", "--processes", str(num_cores), f"fastq/{srr}.fastq"], f"{srr} compressed")

def fetch_srr(srr):
    download_srr(srr)
    dump_fastq(srr)
    compress_fastq(srr)

labels = utils.Label(label_csv)
srr_list = labels.get_srr_list()

executer = utils.Executer()
executer.log(f"SRRs to download: {srr_list}")
executer.run(["mkdir", "-p", "raw", "fastq"], "directories created")

with mp.Pool(processes=batch_size) as pool, tqdm(total=len(srr_list), desc="Fetch Data") as pbar:
    for _ in pool.imap_unordered(fetch_srr, srr_list):
        pbar.update(1)
