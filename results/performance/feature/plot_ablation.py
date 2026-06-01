import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import orjson
import sys

configs_json = sys.argv[1]
configs = orjson.loads(open(configs_json, "r").read())

def plot_performances(model_paths, title, output_plot):
    MAEs = dict()
    for model_name, model_path in model_paths.items():
        predictions_data = pd.read_csv(f"{model_path}/evaluations.csv")
        MAE = predictions_data.loc[predictions_data["Dataset"] == "Test", "MAE"].tolist()
        MAEs[model_name] = MAE

    plt.figure(figsize=(len(MAEs) + 1, 5), dpi=300)
    plt.violinplot(list(MAEs.values()), positions=list(range(len(MAEs))), widths=0.8, showmeans=True)
    plt.xticks(range(len(MAEs)), list(MAEs.keys()), rotation=10, ha="right")
    plt.xlabel("Features Removed")
    plt.axhline(y=np.mean(MAEs["Original"]), color='black', linestyle='--', alpha=0.8, linewidth=1)
    plt.ylabel("MAE")
    plt.title(title)
    plt.savefig(output_plot)

if __name__ == "__main__":
    plot_performances(**configs["Retrained"])
    plot_performances(**configs["Removed"])
