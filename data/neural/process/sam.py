# sam.py

import pysam
from collections import defaultdict

def count_metrics(sam_file):
    sam_data = pysam.AlignmentFile(sam_file, "rb")
    metrics_count = defaultdict(lambda: defaultdict(lambda: {"RD": 0, "ED": 0, "MC": 0}))

    for read in sam_data.fetch():
        flag = read.flag

        if (not (flag & 3 == 3)) or (flag & (4 | 8 | 256 | 512 | 1024 | 2048)):
            continue
            
        rname = sam_data.get_reference_name(read.reference_id)
        pos = read.reference_start + 1
        
        # Process CIGAR string
        indicator = []
        for op, count in read.cigartuples:
            if op == 2:  # D
                indicator.extend(['D'] * count)
            elif op == 7:  # =
                indicator.extend(['='] * count)
            elif op == 8:  # X
                indicator.extend(['X'] * count)

        # Count end depth
        end = pos + len(indicator) - 1 if flag & 16 else pos
        metrics_count[rname][end]["ED"] += 1
        
        # Batch process read depth and mismatches
        for index, char in enumerate(indicator):
            bp = pos + index
            metrics_count[rname][bp]["RD"] += 1
            if char != '=':
                metrics_count[rname][bp]["MC"] += 1

    sam_data.close()

    # Convert to final format
    metrics_data = {}
    for rname, positions in metrics_count.items():
        metrics_entry = {"PS": [], "RD": [], "ED": [], "MC": []}
        for bp, counts in sorted(positions.items()):
            metrics_entry["PS"].append(bp)
            metrics_entry["RD"].append(counts["RD"])
            metrics_entry["ED"].append(counts["ED"])
            metrics_entry["MC"].append(counts["MC"])
        metrics_data[rname] = metrics_entry

    return metrics_data

def count_rtstops(sam_file):
    sam_data = pysam.AlignmentFile(sam_file, "rb")
    rtstops_count = defaultdict(lambda: defaultdict(lambda: {"RD": 0, "ED": 0, "MC": 0}))

    for read in sam_data.fetch():
        flag = read.flag

        if flag:
            continue
            
        rname = sam_data.get_reference_name(read.reference_id)
        pos = read.reference_start + 1
        
        # Process CIGAR string
        indicator = []
        for op, count in read.cigartuples:
            if op == 2:  # D
                indicator.extend(['D'] * count)
            elif op == 7:  # =
                indicator.extend(['='] * count)
            elif op == 8:  # X
                indicator.extend(['X'] * count)

        # Count end depth
        rtstops_count[rname][pos]["ED"] += 1
        
        # Batch process read depth and mismatches
        for index, char in enumerate(indicator):
            bp = pos + index
            rtstops_count[rname][bp]["RD"] += 1
            if char != '=':
                rtstops_count[rname][bp]["MC"] += 1

    sam_data.close()

    # Convert to final format
    rtstops_data = {}
    for rname, positions in rtstops_count.items():
        rtstops_entry = {"PS": [], "RD": [], "ED": [], "MC": []}
        for bp, counts in sorted(positions.items()):
            rtstops_entry["PS"].append(bp)
            rtstops_entry["RD"].append(counts["RD"])
            rtstops_entry["ED"].append(counts["ED"])
            rtstops_entry["MC"].append(counts["MC"])
        rtstops_data[rname] = rtstops_entry

    return rtstops_data
