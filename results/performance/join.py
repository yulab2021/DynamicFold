import pandas as pd
import sys

dataset_csv = sys.argv[1]
outputs_csv = sys.argv[2]
joined_csv = sys.argv[3]

dataset = pd.read_csv(dataset_csv)
outputs = pd.read_csv(outputs_csv)
joined = pd.merge(left=outputs.loc[outputs["Dataset"] == "Test"], right=dataset, left_on="SeqID", right_on="SeqID", how="left")
joined.to_csv(joined_csv, index=False)
