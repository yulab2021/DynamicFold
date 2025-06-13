import sys
import orjson
import pandas as pd
import matplotlib.pyplot as plt

args = orjson.loads(open(sys.argv[1], "r").read())

MAEs = dict()
benchmark_data = pd.read_csv(args["DatasetCSV"])
MAEs["RibonanzaNet"] = benchmark_data["RibonanzaNetMAE"].tolist()

for model_name, evaluations_csv in args["Models"].items():
    predictions_data = pd.read_csv(evaluations_csv)
    MAE = predictions_data["MAE"].tolist()
    MAEs[model_name] = MAE

plt.figure(figsize=(1.5 * len(args["Models"]), 5), dpi=300)
for index, (model_name, MAE) in enumerate(MAEs.items()):
    plt.violinplot(MAE, positions=[index], widths=0.8, showmeans=True)

plt.xticks(range(len(MAEs)), list(MAEs.keys()), rotation=10, ha="right")
plt.ylabel("MAE")
plt.title("Basewise Models")
plt.savefig(args["OutputFigure"])