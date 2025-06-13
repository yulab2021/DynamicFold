import utils
import numpy as np
import orjson
import torch
from torch.utils.data import DataLoader
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sys

configs_json = sys.argv[1]
ref_name = sys.argv[2]
output_basename = sys.argv[3]

def saliency_map(model, batch):
    batch.requires_grad_()
    batch_size, num_channels, seq_len = batch.shape
    outputs = model(batch)
    full_map = torch.zeros(batch_size, num_channels, seq_len, seq_len)
    for b in range(batch_size):
        for o in range(seq_len):
            target = outputs[b, o]
            gradients = torch.autograd.grad(outputs=target, inputs=batch, retain_graph=True, create_graph=False)
            full_map[b, :, o, :] = gradients[0].squeeze(0).detach().cpu()
    return full_map

configs = orjson.loads(open(configs_json, "r").read())
dataset = utils.Dataset(**configs["DatasetArgs"])

test_Xs = [X_entry for index, X_entry in zip(dataset.test_indices, dataset.Xs) if index]
test_Xs = np.concat(test_Xs, axis=1)
sd = np.std(test_Xs, axis=1)

dataset.dataset = dataset.dataset.loc[dataset.dataset["RefName"] == ref_name]
keys, _, Xs, ys = dataset.load(configs["DatasetArgs"]["feature_list"])
dataloader = DataLoader(utils.RNADataset(keys, Xs, ys))

checkpoint = utils.Checkpoint(checkpoint_pt=configs["CheckpointPT"])
model, _ = checkpoint.load(configs["Module"], model_state=True)

for key, inputs, labels in tqdm(dataloader, desc="Calculate Saliency"):
    full_map = saliency_map(model, inputs).squeeze(0).abs().pow(0.5).numpy().max(axis=0)
    plt.figure(figsize=(6, 5), dpi=300)
    plt.imshow(full_map, aspect='auto', cmap=sns.color_palette("Blues", as_cmap=True))
    plt.colorbar(label='$\\sqrt{\\text{Max Normalized Saliency}}$')
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title(f"Saliency Map of {key[0]}")
    plt.savefig(f"{output_basename}_{key[0]}.png")
