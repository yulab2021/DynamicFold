# format.py

import numpy as np
import sys
import sqlite3
import utils
import multiprocessing as mp
from tqdm import tqdm
import orjson
import ast

label_csv = sys.argv[1]
metrics_db = sys.argv[2]
rtstops_db = sys.argv[3]
reference_fasta = sys.argv[4]
alpha = float(sys.argv[5]) # 0.25
strip = ast.literal_eval(sys.argv[6])
dataset_db = sys.argv[7]
table_name = sys.argv[8]
batch_size = int(sys.argv[9])

def onehot_encode(sequence):
    tokens = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'U': [0, 0, 0, 1], 'N': [0.25, 0.25, 0.25, 0.25]}
    encoded = [tokens[nt] for nt in sequence]
    encoded = np.transpose(encoded).tolist()
    return encoded

def winsorize_scale(scores):
    keys = list(scores.keys())
    values = list(scores.values())

    lower_threshold = np.percentile(values, 5)
    upper_threshold = np.percentile(values, 95)

    if lower_threshold == upper_threshold:
        lower_threshold = np.min(values)
        upper_threshold = np.max(values)
        winsorized_values = values
    else:
        winsorized_values = np.clip(values, lower_threshold, upper_threshold)

    if lower_threshold == upper_threshold != 0:
        scaled_values = winsorized_values / upper_threshold
    elif lower_threshold == upper_threshold == 0:
        scaled_values = winsorized_values
    else:
        scaled_values = (winsorized_values - lower_threshold) / (upper_threshold - lower_threshold)

    results = {key: float(value) for key, value in zip(keys, scaled_values)}
    return results

def filter_entries(keys):
    ref_names = dict()
    sample_list = labels.unique_values("Sample")
    for sample in sample_list:
        ref_names[sample] = dict()
        srr_list = labels.get_srr_list({"Sample": [sample]})
        for srr in srr_list:
            ref_names[sample][srr] = list()

    for key in keys:
        sample = key.split("|")[0]
        srr = key.split("|")[3]
        ref_name = key.split("|")[4]
        ref_names[sample][srr].append(ref_name)

    results = dict()
    for sample in ref_names:
        results[sample] = list(set.intersection(*[set(ref_names[sample][srr]) for srr in ref_names[sample]]))

    return results

def read_fasta(filename):
    sequences = {}
    with open(filename, 'r') as file:
        current_seq_id = None
        sequence_data = ""
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line.startswith('>'):
                if current_seq_id:  # Store previous sequence
                    sequences[current_seq_id] = sequence_data
                current_seq_id = line[1:].split(' ')[0]  # Extract ID from header
                sequence_data = ""
            else:
                sequence_data += line
        if current_seq_id:  # Store last sequence
            sequences[current_seq_id] = sequence_data
    return sequences

def load_reactivity(sample, ref_name):
    NAIN3_list = labels.get_srr_list({"Sample": [sample], "Experiment": ["icSHAPE"], "Group": ["NAIN3"]})
    DMSO_list = labels.get_srr_list({"Sample": [sample], "Experiment": ["icSHAPE"], "Group": ["DMSO"]})

    DMSO_depth = dict()
    DMSO_stop = dict()
    NAIN3_stop = dict()

    num_rep = len(DMSO_list)
    for srr in DMSO_list:
        key = f"{sample}|icSHAPE|DMSO|{srr}|{ref_name}"
        rtstops_entry = rtstops.read(key)
        for index, pos in enumerate(rtstops_entry["PS"]):
            if pos not in DMSO_depth:
                DMSO_depth[pos] = 0
            if pos not in DMSO_stop:
                DMSO_stop[pos] = 0
            DMSO_depth[pos] += rtstops_entry["RD"][index] / num_rep
            DMSO_stop[pos] += rtstops_entry["ED"][index] / num_rep

    num_rep = len(NAIN3_list)
    for srr in NAIN3_list:
        key = f"{sample}|icSHAPE|NAIN3|{srr}|{ref_name}"
        rtstops_entry = rtstops.read(key)
        for index, pos in enumerate(rtstops_entry["PS"]):
            if pos not in NAIN3_stop:
                NAIN3_stop[pos] = 0
            NAIN3_stop[pos] += rtstops_entry["ED"][index] / num_rep

    start = min(min(DMSO_stop.keys()), min(NAIN3_stop.keys())) # 1-based position
    end = max(max(DMSO_stop.keys()), max(NAIN3_stop.keys())) # 1-based position
        
    md_indicators = dict() # missing data indicators, 0b001: DMSO missing, 0b010: NAIN3 missing, 0b100: RNA-Seq missing
    reactivity_scores = dict()
    for pos in range(start, end + 1):
        md_indicators[pos] = 0
        reactivity_scores[pos] = 0
        if pos not in DMSO_stop:
            md_indicators[pos] |= 0b001
        if pos not in NAIN3_stop:
            md_indicators[pos] |= 0b010
        if md_indicators[pos] == 0:
            reactivity_scores[pos] = (NAIN3_stop[pos] - alpha * DMSO_stop[pos]) / DMSO_depth[pos]

    total_density = np.sum(list(DMSO_depth.values()))
    reactivity_scores = winsorize_scale(reactivity_scores)

    return reactivity_scores, md_indicators, total_density

def load_metrics(sample, ref_name):
    srr_list = labels.get_srr_list({"Sample": [sample], "Experiment": ["RNA-Seq"]})
    
    keys = list()
    for srr in srr_list:
        key = f"{sample}|RNA-Seq|NA|{srr}|{ref_name}"
        if key in metrics_keys:
            keys.append(key)
    
    num_rep = len(keys)
    if num_rep == 0:
        raise ValueError("no RNA-Seq data")
    
    read_depth = dict()
    end_depth = dict()
    end_rate = dict()
    mismatch_count = dict()
    mismatch_rate = dict()

    for key in keys:
        metrics_entry = metrics.read(key)
        for index, pos in enumerate(metrics_entry["PS"]):
            if pos not in read_depth:
                read_depth[pos] = 0
            if pos not in end_depth:
                end_depth[pos] = 0
            if pos not in mismatch_count:
                mismatch_count[pos] = 0
            read_depth[pos] += metrics_entry["RD"][index] / num_rep
            end_depth[pos] += metrics_entry["ED"][index] / num_rep
            mismatch_count[pos] += metrics_entry["MC"][index] / num_rep

    for pos in read_depth:
        end_rate[pos] = end_depth[pos] / read_depth[pos]
        mismatch_rate[pos] = mismatch_count[pos] / read_depth[pos]

    total_depth = np.sum(list(read_depth.values()))
    total_end_rate = np.sum(list(end_rate.values()))
    total_mismatch_rate = np.sum(list(mismatch_rate.values()))
    read_depth = winsorize_scale(read_depth)

    return read_depth, end_rate, mismatch_rate, total_depth, total_end_rate, total_mismatch_rate

def format_entry(args):
    sample, ref_name = args
    entry_name = f"{sample}|{ref_name}"

    try:
        reactivity_scores_dict, reactivity_indicators_dict, total_density = load_reactivity(sample, ref_name)
        read_depth_dict, end_rate_dict, mismatch_rate_dict, total_depth, total_end_rate, total_mismatch_rate = load_metrics(sample, ref_name)
    except ValueError as e:
        executer.log(f"{e} for {entry_name}")
        return None
    
    sequence = reference_transcriptome[ref_name].replace("T", "U")
    full_length = len(sequence)
    channel_A, channel_C, channel_G, channel_U = onehot_encode(sequence)
    
    indicators = [0] * full_length # 0b001: DMSO missing, 0b010: NAIN3 missing, 0b100: RNA-Seq missing
    reactivity = [0] * full_length
    read_depth = [0] * full_length
    end_rate = [0] * full_length
    mismatch_rate = [0] * full_length
    for pos in range(1, full_length + 1):
        if pos in reactivity_scores_dict:
            reactivity[pos - 1] = reactivity_scores_dict[pos]
            indicators[pos - 1] |= reactivity_indicators_dict[pos]
        else:
            indicators[pos - 1] |= 0b011
        if pos in read_depth_dict:
            read_depth[pos - 1] = read_depth_dict[pos]
            end_rate[pos - 1] = end_rate_dict[pos]
            mismatch_rate[pos - 1] = mismatch_rate_dict[pos]
        else:
            indicators[pos - 1] |= 0b100

    # find the first and last pos where indicator == 0
    start = None # 1-based position
    end = None # (-1)-based position in reverse
    for index in range(full_length):
        if indicators[index] == 0 and start is None:
            start = index + 1
        if indicators[index] == 0:
            end = index - full_length
            
    if (start is None) or (end is None) or (start - 1 >= end + full_length):
        executer.log(f"no valid data for {entry_name}")
        return None
    
    valid_length = end + full_length - start + 2
    strip_length = full_length - valid_length
    gap = 0
    for index in range(start - 1, end + full_length + 1):
        if indicators[index] != 0:
            gap += 1
    mean_depth = total_depth / valid_length
    mean_end = total_end_rate / valid_length
    mean_density = total_density / valid_length
    mean_mismatch = total_mismatch_rate / valid_length

    if strip:
        sequence = sequence[(start - 1):(end + full_length + 1)]
        channel_A = channel_A[(start - 1):(end + full_length + 1)]
        channel_C = channel_C[(start - 1):(end + full_length + 1)]
        channel_G = channel_G[(start - 1):(end + full_length + 1)]
        channel_U = channel_U[(start - 1):(end + full_length + 1)]
        read_depth = read_depth[(start - 1):(end + full_length + 1)]
        end_rate = end_rate[(start - 1):(end + full_length + 1)]
        mismatch_rate = mismatch_rate[(start - 1):(end + full_length + 1)]
        reactivity = reactivity[(start - 1):(end + full_length + 1)]
        indicators = indicators[(start - 1):(end + full_length + 1)]

    entry = (entry_name, 
             orjson.dumps(channel_A), orjson.dumps(channel_C), orjson.dumps(channel_G), orjson.dumps(channel_U), orjson.dumps(read_depth), orjson.dumps(end_rate), orjson.dumps(mismatch_rate), orjson.dumps(reactivity), orjson.dumps(indicators), 
             sample, ref_name, sequence, start, end, full_length, valid_length, strip_length,
             mean_depth, mean_end, mean_density, mean_mismatch, gap
            )
    return entry

labels = utils.Label(label_csv)
reference_transcriptome = read_fasta(reference_fasta)

metrics = utils.Database(metrics_db, "metrics")
rtstops = utils.Database(rtstops_db, "rtstops")
executer = utils.Executer()

metrics.connect()
rtstops.connect()
metrics_keys = metrics.list()
rtstops_keys = rtstops.list()
ref_names = filter_entries(rtstops_keys + metrics_keys)
args = [(sample, ref_name) for sample in ref_names for ref_name in ref_names[sample]]

dataset = sqlite3.connect(dataset_db)
dataset_cursor = dataset.cursor()
dataset_cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (SeqID TEXT PRIMARY KEY, A TEXT, C TEXT, G TEXT, U TEXT, RD TEXT, ER TEXT, MR TEXT, RT TEXT, IC TEXT, Sample TEXT, RefName TEXT, Sequence TEXT, Start INT, End INT, FullLength INT, ValidLength INT, StripLength INT, MeanDepth REAL, MeanEnd REAL, MeanDensity REAL, MeanMismatch REAL, Gap INT)")

with mp.Pool(processes=batch_size) as pool:
    for entry in tqdm(pool.imap_unordered(format_entry, args), total=len(args), desc="Format Dataset"):
        if entry is not None:
            dataset_cursor.execute(f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", entry)

dataset.commit()
dataset.close()
metrics.close()
rtstops.close()
