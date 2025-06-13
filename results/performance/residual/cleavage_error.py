import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys
import orjson

data_csv = sys.argv[1]
bootstrap_size = int(sys.argv[2])
eps = float(sys.argv[3])
levels = int(sys.argv[4])
grid_size = int(sys.argv[5])
output_plot = sys.argv[6]

data = pd.read_csv(data_csv)
dynamic_error = list()
end_rate = list()
for _, row in data.iterrows():
    reactivity = np.array(orjson.loads(row["RT"]))
    dynamic_prediction = np.array(orjson.loads(row["Predictions"]))
    dynamic_error.extend((reactivity - dynamic_prediction).tolist())
    end_rate.extend(orjson.loads(row["ER"]))

indices = list(range(len(dynamic_error)))
indices = np.random.choice(indices, size=bootstrap_size, replace=True)
error_sample = np.array(dynamic_error)[indices]
end_rate_sample = np.log10(np.array(end_rate)[indices] + eps)

plt.figure(figsize=(6, 5), dpi=300)
sns.kdeplot(x=end_rate_sample, y=error_sample, fill=True, levels=levels, cbar=True, cmap=sns.color_palette("Blues", as_cmap=True), gridsize=grid_size)
plt.xlim((np.log10(eps), 0))
plt.ylim((-1, 1))
plt.xlabel(f"$\\log_{{10}}$(Cleavage Rate + {eps})")
plt.ylabel("DynamicFold Error")
plt.title("Distribution of Cleavage Rate vs. Error per Base")
plt.savefig(output_plot)
