# rnaseq.py

import multiprocessing as mp
import subprocess
from tqdm import tqdm
import sys
import pandas as pd
import ast
import datetime
import sam
import utils

input_dir = sys.argv[1]
label_csv = sys.argv[2]
bowtie2_index = sys.argv[3]
reference_fasta = sys.argv[4]
num_cores = int(sys.argv[5])
batch_size = int(sys.argv[6])

logs = open("logs.txt", "a")

def dump_fastq(sample_name):
    subprocess.run(["fasterq-dump", "--force", "--verbose", "--split-files", "--outdir", f"cache/{sample_name}/", "--temp", "cache/", "--threads", str(num_cores), f"{input_dir}/{sample_name}.sra"], stdout=logs, stderr=logs)

def trim_adaptor(sample_name):
    subprocess.run(["trim_galore", "--trim-n", "--output_dir", f"cache/{sample_name}/", "--basename", sample_name, "--cores", str(num_cores), "--paired", f"cache/{sample_name}/{sample_name}_1.fastq", f"cache/{sample_name}/{sample_name}_2.fastq"], stdout=logs, stderr=logs)

def align_reads(sample_name):
    subprocess.run(["bowtie2", "--xeq", "--non-deterministic", "--end-to-end", "--very-sensitive", "--threads", str(num_cores), "-x", bowtie2_index, "-1", f"cache/{sample_name}/{sample_name}_val_1.fq", "-2", f"cache/{sample_name}/{sample_name}_val_2.fq", "-S", f"cache/{sample_name}/{sample_name}.sam"], stdout=logs, stderr=logs)

def process_sam(sample_name):
    metrics_data = sam.count_metrics(f"cache/{sample_name}/{sample_name}.sam")
    return metrics_data

def process_sample(sample_name):
    subprocess.run(["mkdir", f"cache/{sample_name}/"], stdout=logs, stderr=logs)

    dump_fastq(sample_name)
    logs.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {sample_name} dumped\n")
    logs.flush()

    trim_adaptor(sample_name)
    logs.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {sample_name} trimmed\n")
    logs.flush()

    align_reads(sample_name)
    logs.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {sample_name} aligned\n")
    logs.flush()

    metrics_data = process_sam(sample_name)
    return sample_name, metrics_data

def write_database(database, labels, sample_name, output_data):
    base_name = utils.get_base_name(labels, sample_name)
    database.connect()
    for ref_name, data in output_data.items():
        table_name = f"{base_name}|{ref_name}"
        database.write(table_name, data)
    database.close()

    logs.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {sample_name} processed\n")
    logs.flush()

    subprocess.run(["rm", "-r", f"cache/{sample_name}"], stdout=logs, stderr=logs)

labels = utils.parse_labels(label_csv)
label_data = pd.read_csv(label_csv, keep_default_na=False, na_values=[])
sample_list = label_data[label_data["Experiment"] == "RNA-Seq"]["SRR"].to_list()
sample_list = [srr for item in sample_list for srr in ast.literal_eval(item)]
subprocess.run(["mkdir", "cache"], stdout=logs, stderr=logs)
metrics = utils.Database("metrics.db", "metrics")

logs.write(f"Samples to process: {sample_list}\n")
logs.flush()

with mp.Pool(processes=batch_size) as pool:
    for sample_name, output_data in tqdm(pool.imap(process_sample, sample_list), total=len(sample_list), desc="Process Sample"):
        write_database(metrics, labels, sample_name, output_data)