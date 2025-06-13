import utils
import numpy as np
import orjson
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

configs_json = sys.argv[1]
output_plot = sys.argv[2]

configs = orjson.loads(open(configs_json, "r").read())
dataset = utils.Dataset(**configs["DatasetArgs"])
checkpoint = utils.Checkpoint(checkpoint_pt=configs["CheckpointPT"])
model, _ = checkpoint.load(configs["Module"], model_state=True)
criterion = torch.nn.L1Loss()

mean_saliencies = list()
for _, inputs, labels in tqdm(dataset.test_dataloader, desc="Calculate Saliency"):
    inputs.requires_grad_()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    gradients = torch.autograd.grad(outputs=loss, inputs=inputs)
    mean_saliencies.append(gradients[0].abs().squeeze(0).detach().cpu().mean(dim=1).numpy())

test_Xs = [X_entry for index, X_entry in zip(dataset.test_indices, dataset.Xs) if index]
test_Xs = np.concat(test_Xs, axis=1)
sd = np.std(test_Xs, axis=1)
mean_saliencies = np.array(mean_saliencies)
normalized_saliencies = np.log10(mean_saliencies * sd).T.tolist()
labels = ["A", "C", "G", "U", "Read Depth", "Cleavage Rate", "Mismatch Rate", "RibonanzaNet"]

plt.figure(figsize=(10, 5), dpi=300)
plt.violinplot(normalized_saliencies, positions=list(range(len(labels))), widths=0.8, showmeans=True)
plt.xticks(range(8), labels, rotation=10, ha="right")
plt.ylabel("$\\log_{10}$(Normalized Saliency)")
plt.title("Distribution of Normalized Feature Saliency per Sequence")
plt.savefig(output_plot)
