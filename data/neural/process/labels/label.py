# label.py

import pandas as pd
import sys

def label_experiment(source):
    data = {"GSM": [], "SRR": [], "Sample": [], "Experiment": [], "Group": [], "Repetition": [], "Layout": [], "Instrument": []}
    for index, row in source.iterrows():
        data["GSM"].append(row["SampleName"])
        data["SRR"].append(row["Run"])
        data["Layout"].append(row["LibraryLayout"])
        data["Instrument"].append(row["Instrument"])
        label = row["Experiment Title"]
        
        # Sample
        if "D0" in label:
            data["Sample"].append("D0")
        elif "D7" in label:
            data["Sample"].append("D7")
        elif "D8" in label:
            data["Sample"].append("D8")
        elif "D14" in label:
            data["Sample"].append("D14")
        else:
            data["Sample"].append("OTHER")

        # Experiment & Group
        if "NAIN3" in label:
            data["Experiment"].append("icSHAPE")
            data["Group"].append("NAIN3")
        elif "DMSO" in label:
            data["Experiment"].append("icSHAPE")
            data["Group"].append("DMSO")
        elif "RNASeq" in label:
            data["Experiment"].append("RNA-Seq")
            data["Group"].append("NA")
        else:
            data["Experiment"].append("OTHER")
            data["Group"].append("OTHER")

        # Repetition
        if "rep1" in label:
            data["Repetition"].append("rep1")
        elif "rep2" in label:
            data["Repetition"].append("rep2")
        else:
            data["Repetition"].append("NA")
    
    # Remove OTHER rows
    df = pd.DataFrame(data)
    filtered = df[~df.isin(['OTHER']).any(axis=1)]
    return filtered

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    source = pd.read_csv(input_file)
    
    df = label_experiment(source)
    df.to_csv(output_file, index=False)