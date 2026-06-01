import utils
import numpy as np
import orjson
import torch
from tqdm import tqdm
import pandas as pd
import sys

configs_json = sys.argv[1]

configs = orjson.loads(open(configs_json, "r").read())
dataset = utils.Dataset(**configs["DatasetArgs"])
checkpoint = utils.Checkpoint(checkpoint_pt=configs["CheckpointPT"])
model, _ = checkpoint.load(configs["Module"], model_state=True)
criterion = torch.nn.L1Loss()

test_Xs = [X_entry for index, X_entry in zip(dataset.indices["Test"], dataset.Xs) if index]
test_Xs = np.concat(test_Xs, axis=1)
sd = np.std(test_Xs, axis=1)
name_map = {"A": "A", "C": "C", "G": "G", "U": "U", "RD": "Read Depth", "ER": "Cleavage Rate", "MR": "Mismatch Rate"}
features = [name_map[k] for k in configs["DatasetArgs"]["feature_list"]]

model.eval()
for name, module in model.named_modules():
    if "gru" in name.lower():
        module.train()

mean_saliencies = dict()
max_saliencies = dict()
for key, inputs, labels in tqdm(dataset.dataloaders["Test"], desc="Calculate Saliency"):
    inputs.requires_grad_()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    gradients = torch.autograd.grad(outputs=loss, inputs=inputs)
    mean_saliency = gradients[0].squeeze(0).abs().mean(dim=1).detach().cpu().numpy() * sd
    mean_saliencies[key] = dict(zip(features, mean_saliency.tolist()))
    max_saliency = torch.max(gradients[0].squeeze(0).abs(), dim=1).values.detach().cpu().numpy() * sd
    max_saliencies[key] = dict(zip(features, max_saliency.tolist()))

mean_saliencies = pd.DataFrame(mean_saliencies).transpose()
mean_saliencies.index.name = "SeqID"
mean_saliencies.to_csv("mean_saliencies.csv")

max_saliencies = pd.DataFrame(max_saliencies).transpose()
max_saliencies.index.name = "SeqID"
max_saliencies.to_csv("max_saliencies.csv")
