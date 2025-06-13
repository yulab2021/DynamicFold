import pandas as pd
import matplotlib.pyplot as plt
import orjson
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--configs", "-c", type=str, required=True, help="Path to the JSON configuration file.")
args = parser.parse_args()

configs = orjson.loads(open(args.configs, "r").read())

def plot_performances(dataset_csv, evaluations_csvs, dataset_types, title, output_dir):
    MAEs = dict()
    benchmark_data = pd.read_csv(dataset_csv)
    MAEs["RibonanzaNet"] = benchmark_data["RibonanzaNetMAE"].tolist()

    for dataset_type in dataset_types:
        for model_name, evaluations_csv in evaluations_csvs.items():
            predictions_data = pd.read_csv(evaluations_csv)
            if "Dataset" in predictions_data.columns:
                MAE = predictions_data.loc[predictions_data["Dataset"] == dataset_type, "MAE"].tolist()
            elif dataset_type == "Test":
                MAE = predictions_data["MAE"].tolist()
            else:
                continue
            MAEs[model_name] = MAE

        plt.figure(figsize=(len(MAEs) + 1, 5), dpi=300)
        plt.violinplot(list(MAEs.values()), positions=list(range(len(MAEs))), widths=0.8, showmeans=True)
        plt.xticks(range(len(MAEs)), list(MAEs.keys()))
        plt.ylabel("MAE")
        plt.title(f"{title} [{dataset_type} Set]")
        plt.savefig(f"{output_dir}/{title.lower()}_{dataset_type.lower()}_perf.png")

def plot_losses(report_jsons, loss_types, title, output_dir):
    losses = {loss_type: dict() for loss_type in loss_types}

    for model_name, loss_json in report_jsons.items():
        report_data = orjson.loads(open(loss_json, "r").read())
        for loss_type in loss_types:
            losses[loss_type][model_name] = report_data[loss_type]
    
    for loss_type, model_losses in losses.items():
        plt.figure(figsize=(6, 5), dpi=300)
        for model_name, loss in model_losses.items():
            plt.plot(range(len(loss)), loss, label=model_name)
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{title} [{loss_type}]")
        plt.savefig(f"{output_dir}/{title.lower()}_{loss_type.lower()}.png")

if __name__ == "__main__":
    plot_performances(**configs["PerformanceArgs"])
    plot_losses(**configs["LossArgs"])
