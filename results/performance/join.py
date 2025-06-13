import pandas as pd
import sys

dataset_csv = sys.argv[1]
evaluations_csv = sys.argv[2]
output_csv = sys.argv[2]

dataset = pd.read_csv(dataset_csv)
evaluations = pd.read_csv(evaluations_csv)
joined = pd.merge(left=evaluations.loc[evaluations["Dataset"] == "Test"], right=dataset, left_on="SeqID", right_on="SeqID", how="left")
joined.to_csv(output_csv, index=False)
