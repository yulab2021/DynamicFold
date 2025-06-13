# rnaseq.py

import multiprocessing as mp
from tqdm import tqdm
import sys
import sam
import utils

input_dir = sys.argv[1]
label_csv = sys.argv[2]
bowtie2_index = sys.argv[3]
reference_fasta = sys.argv[4]
num_cores = int(sys.argv[5])
batch_size = int(sys.argv[6])

def decompress_fastq(srr):
    executer.run(["cp", f"{input_dir}/{srr}_1.fastq.gz", f"{input_dir}/{srr}_2.fastq.gz", f"cache/{srr}/"], f"{srr} copied")
    executer.run(["pigz", "--decompress", "--processes", str(num_cores), f"cache/{srr}/{srr}_1.fastq.gz", f"cache/{srr}/{srr}_2.fastq.gz"], f"{srr} decompressed")

def trim_adaptor(srr):
    executer.run(["trim_galore", "--trim-n", "--output_dir", f"cache/{srr}/", "--basename", srr, "--cores", str(num_cores), "--paired", f"cache/{srr}/{srr}_1.fastq", f"cache/{srr}/{srr}_2.fastq"], f"{srr} trimmed")

def align_reads(srr):
    executer.run(["bowtie2", "--xeq", "--non-deterministic", "--end-to-end", "--very-sensitive", "--threads", str(num_cores), "-x", bowtie2_index, "-1", f"cache/{srr}/{srr}_val_1.fq", "-2", f"cache/{srr}/{srr}_val_2.fq", "-S", f"cache/{srr}/{srr}.sam"], f"{srr} aligned")

def process_sam(srr):
    metrics_data = sam.count_metrics(f"cache/{srr}/{srr}.sam")
    return metrics_data

def process_srr(srr):
    executer.run(["mkdir", "-p", f"cache/{srr}/"], f"cache/{srr}/ created")

    decompress_fastq(srr)
    trim_adaptor(srr)
    align_reads(srr)

    metrics_data = process_sam(srr)
    return srr, metrics_data

def write_database(srr, output_data):
    base_name = labels.get_base_name(srr)
    metrics.connect()
    for ref_name, data in output_data.items():
        key = f"{base_name}|{ref_name}"
        metrics.write(key, data)
    metrics.close()

    executer.log(f"{srr} processed")
    executer.run(["rm", "-r", f"cache/{srr}"], f"cache/{srr}/ removed")

labels = utils.Label(label_csv)
srr_list = labels.get_srr_list({"Experiment": ["RNA-Seq"]})

metrics = utils.Database("metrics.db", "metrics")
executer = utils.Executer()
executer.log(f"SRRs to process: {srr_list}")

with mp.Pool(processes=batch_size) as pool:
    for srr, output_data in tqdm(pool.imap_unordered(process_srr, srr_list), total=len(srr_list), desc="Process SRRs"):
        write_database(srr, output_data)
