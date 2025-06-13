# label.py

import pandas as pd
import sys

def label_experiment(source):
    data = {"GSM": [], "SRR": [], "Sample": [], "Experiment": [], "Group": [], "Repetition": [], "Layout": []}
    for index, row in source.iterrows():
        data["GSM"].append(row["SampleName"])
        data["SRR"].append(row["Run"])
        data["Layout"].append(row["LibraryLayout"])
        label = row["Experiment Title"]
        
        # Sample
        if (("sphere" in label) or ("Sphere" in label) or ("4h" in label)) and (("Elavl1a" in label) or ("elavl1a" in label)):
            data["Sample"].append("4h-mut")
        elif ("sphere" in label) or ("Sphere" in label) or ("4h" in label):
            data["Sample"].append("4h-wt")
        elif (("shield" in label) or ("Shield" in label) or ("6h" in label)) and (("Elavl1a" in label) or ("elavl1a" in label)):
            data["Sample"].append("6h-mut")
        elif ("shield" in label) or ("Shield" in label) or ("6h" in label):
            data["Sample"].append("6h-wt")
        elif ("2h" in label) and (("Elavl1a" in label) or ("elavl1a" in label)):
            data["Sample"].append("2h-mut")
        elif "2h" in label:
            data["Sample"].append("2h-wt")
        else:
            data["Sample"].append("OTHER")

        # Experiment & Group
        if "NAIN3" in label:
            data["Experiment"].append("icSHAPE")
            data["Group"].append("NAIN3")
        elif "DMSO" in label:
            data["Experiment"].append("icSHAPE")
            data["Group"].append("DMSO")
        elif ("RNA-Seq" in label) or ("rna-seq" in label):
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
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    source = pd.read_csv(input_file)
    
    df = label_experiment(source)
    df.to_csv(output_file, index=False)