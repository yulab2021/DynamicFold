import utils
import numpy as np
import orjson
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import sys

configs_json = sys.argv[1]
bootstrap = int(sys.argv[2])
num_components = int(sys.argv[3])
plot_components = int(sys.argv[4])
output_basename = sys.argv[5]

configs = orjson.loads(open(configs_json, "r").read())
dataset = utils.Dataset(**configs["DatasetArgs"])
checkpoint = utils.Checkpoint(checkpoint_pt=configs["CheckpointPT"])
model, _ = checkpoint.load(configs["Module"], model_state=True)
cmap_points = LinearSegmentedColormap.from_list("BlueRed", ['#0000FF', '#FF0000'], N=256)
group_colors = {b: cm.viridis(i / 3) for i, b in enumerate("ACGU")}

def onehot_decode(x):
    decoded = list()
    for base in x:
        if base == [1, 0, 0, 0]:
            decoded.append("A")
        elif base == [0, 1, 0, 0]:
            decoded.append("C")
        elif base == [0, 0, 1, 0]:
            decoded.append("G")
        elif base == [0, 0, 0, 1]:
            decoded.append("U")
        else:
            raise ValueError("invalid base encoding")
    return decoded

hidden_states = list()
base_type = list()
reactivity = list()
for _, inputs, labels in tqdm(dataset.test_dataloader, desc="Compute States"):
    outputs = model(inputs, unembed=False, logits=True)
    reactivity.extend(labels.squeeze(0).cpu().tolist())
    base_type.extend(onehot_decode(inputs.permute(0, 2, 1).squeeze(0).cpu()[:,:4].tolist()))
    hidden_states.extend(outputs.squeeze(0).detach().cpu().tolist())

hidden_states = np.array(hidden_states)
reactivity = np.array(reactivity)
base_type = np.array(base_type)
indices = list(range(len(reactivity)))
sample_indices = np.random.choice(indices, bootstrap, replace=True)
hidden_states = hidden_states[sample_indices]
reactivity = reactivity[sample_indices]
base_type = base_type[sample_indices]

reducer = PCA(n_components=num_components)
hidden_states = reducer.fit_transform(hidden_states)

colors = [group_colors[b] for b in base_type]
for c in range(0, plot_components, 2):
    plt.figure(figsize=(6, 5), dpi=300)
    scatter = plt.scatter(hidden_states[:,c], hidden_states[:,c+1], s=1, c=reactivity, marker="o", cmap=cmap_points)
    plt.xlabel(f"Component {c + 1}")
    plt.ylabel(f"Component {c + 2}")
    plt.colorbar(scatter, label="Reactivity")
    plt.title("PCA Decomposition of Hidden States")
    plt.savefig(f"{output_basename}_reactivity_{c + 1}_{c + 2}.png")

    plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter(hidden_states[:,c], hidden_states[:,c+1], s=1, c=colors)
    handles = [plt.scatter([], [], color=group_colors[b], label=b) for b in group_colors]
    plt.legend(handles=handles)
    plt.xlabel(f"Component {c + 1}")
    plt.ylabel(f"Component {c + 2}")
    plt.title("PCA Decomposition of Hidden States")
    plt.savefig(f"{output_basename}_base_{c + 1}_{c + 2}.png", bbox_inches='tight')

