import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd

dataset_csv = sys.argv[1]
output_plot = sys.argv[2]

dataset = pd.read_csv(dataset_csv)
data = {
    "Value": pd.concat([dataset["Start"] - 1, -dataset["End"] - 1]),
    "Location": ["5'"] * len(dataset["Start"]) + ["3'"] * len(dataset["End"])
}
plt.figure(figsize=(5, 5), dpi=300)
sns.histplot(data, x="Value", hue="Location", element="step")
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.xlabel("Number of Missing Bases")
plt.ylabel("Frequency")
plt.title("Distribution of Missing Data per Sequence")
plt.tight_layout()
plt.savefig(output_plot)
