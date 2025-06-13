# icshape.py

import sys
from tqdm import tqdm
import multiprocessing as mp
import sam
import utils

input_dir = sys.argv[1]
label_csv = sys.argv[2]
trim_head = sys.argv[3] # 15
bowtie2_index = sys.argv[4]
reference_fasta = sys.argv[5]
num_cores = int(sys.argv[6])
batch_size = int(sys.argv[7])

def decompress_fastq(srr):
    executer.run(["cp", f"{input_dir}/{srr}.fastq.gz", f"cache/{srr}/"], f"{srr} copied")
    executer.run(["pigz", "--decompress", "--processes", str(num_cores), f"cache/{srr}/{srr}.fastq.gz"], f"{srr} decompressed")

def trim_adaptor(srr):
    executer.run(["trim_galore", "--trim-n", "--output_dir", f"cache/{srr}/", "--basename", srr, "--cores", str(num_cores), f"cache/{srr}/{srr}.fastq"], f"{srr} trimmed")

def collapse_reads(srr):
    executer.run(["clumpify.sh", f"in=cache/{srr}/{srr}_trimmed.fq", f"out=cache/{srr}/{srr}_deduped.fq", f"threads={str(num_cores)}", "dedupe=t", "subs=0", "usetmpdir=t", "tmpdir=cache/"], f"{srr} collapsed")

def remove_index(srr):
    executer.run(["cutadapt", "--cut", trim_head, "--cores", str(num_cores), "--output", f"cache/{srr}/{srr}_unindex.fq", f"cache/{srr}/{srr}_deduped.fq"], f"{srr} index removed")

def align_reads(srr):
    executer.run(["bowtie2", "--xeq", "--non-deterministic", "--end-to-end", "--very-sensitive", "--threads", str(num_cores), "-x", bowtie2_index, "-U", f"cache/{srr}/{srr}_unindex.fq", "-S", f"cache/{srr}/{srr}.sam"], f"{srr} aligned")

def process_sam(srr):
    rtstops_data = sam.count_rtstops(f"cache/{srr}/{srr}.sam")
    return rtstops_data

def process_srr(srr):
    executer.run(["mkdir", "-p", f"cache/{srr}/"], f"cache/{srr}/ created")

    decompress_fastq(srr)
    trim_adaptor(srr)
    collapse_reads(srr)
    remove_index(srr)
    align_reads(srr)

    rtstops_data = process_sam(srr)
    return srr, rtstops_data

def write_database(srr, output_data):
    base_name = labels.get_base_name(srr)
    rtstops.connect()
    for ref_name, data in output_data.items():
        key = f"{base_name}|{ref_name}"
        rtstops.write(key, data)
    rtstops.close()

    executer.log(f"{srr} processed")
    executer.run(["rm", "-r", f"cache/{srr}"], f"cache/{srr}/ removed")

labels = utils.Label(label_csv)
srr_list = labels.get_srr_list({"Experiment": ["icSHAPE"]})

rtstops = utils.Database("rtstops.db", "rtstops")
executer = utils.Executer()
executer.log(f"SRRs to process: {srr_list}")

with mp.Pool(processes=batch_size) as pool:
    for srr, output_data in tqdm(pool.imap_unordered(process_srr, srr_list), total=len(srr_list), desc="Process SRRs"):
        write_database(srr, output_data)