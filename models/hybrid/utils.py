from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from tqdm import tqdm
import numpy as np
import pandas as pd
import io
import hashlib
import importlib
import datetime
import orjson
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
import torch.nn.functional as F

class DeviceManager:
    def __init__(self):
        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
    
    def get(self):
        return self.device
    
    def set(self, device:str):
        self.device = torch.device(device)

dm = DeviceManager()

def extend_dict(this:dict[str,list], other:dict[str,list]):
    if set(this.keys()) != set(other.keys()):
        raise ValueError("non-conformable dict")
    if not (all(isinstance(value, list) for value in this.values()) and all(isinstance(value, list) for value in other.values())):
        raise ValueError("invalid dict format")

    results = dict()
    for key in this.keys():
        results[key] = list()
        results[key].extend(this[key])
        results[key].extend(other[key])
    
    return results

def permute_dict(original:dict):
    keys = original.keys()
    values = original.values()
    permuted = [dict(zip(keys, v)) for v in zip(*values)]
    return permuted

def noisy_argsort(sequence:list[int], noise_scale:float):
    sequence = np.array(sequence) + np.random.uniform(-noise_scale, noise_scale, len(sequence))
    return np.argsort(sequence)

def collate_batch(batch:list[tuple[torch.Tensor|str]]):
    keys, Xs, ys = zip(*batch)
    max_len = max([y.shape[0] for y in ys])
    Xs = [F.pad(X, (0, max_len - X.shape[1]), "constant", 0) for X in Xs]
    ys = [F.pad(y, (0, max_len - y.shape[0]), "constant", 0) for y in ys]
    Xs = torch.stack(Xs)
    ys = torch.stack(ys)
    return keys, Xs, ys

class RNADataset(utils.Dataset):
    def __init__(self, keys:list[str], Xs:list[np.ndarray], ys:list[np.ndarray]):
        assert len(keys) == len(Xs) == len(ys)
        self.keys = keys
        self.Xs = Xs
        self.ys = ys

    def __len__(self):
        return len(self.keys)
    
    def entry_lengths(self):
        return [y.shape[0] for y in self.ys]

    def __getitem__(self, index:int):
        key = self.keys[index]
        X = torch.tensor(self.Xs[index], dtype=torch.float32, device=dm.device)
        y = torch.tensor(self.ys[index], dtype=torch.float32, device=dm.device)
        return key, X, y

class LengthAwareSampler(utils.Sampler):
    def __init__(self, dataset:RNADataset, batch_size:int, noise_scale:float):
        self.entry_lengths = dataset.entry_lengths()
        self.batch_size = batch_size
        self.noise_scale = noise_scale
        self.len = (len(self.entry_lengths) + batch_size - 1) // batch_size
       
    def __iter__(self):
        indices = noisy_argsort(self.entry_lengths, self.noise_scale)
        batches = [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]
        np.random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return self.len

class Dataset:
    def __init__(self, dataset_csv:str, bootstrap:int, feature_list:list[str], val_size:float, test_size:float, seed:int, batch_size:int, noise_scale:float):
        self.dataset = pd.read_csv(dataset_csv)
        if bootstrap > 0:
            self.dataset = self.dataset.sample(n=bootstrap, replace=True)
        self.load(feature_list)
        self.split(val_size, test_size, seed, batch_size, noise_scale)

    def load(self, feature_list:list[str]):
        self.keys = list()
        self.ref_names = list()
        self.Xs = list()
        self.ys = list()

        for _, row in self.dataset.iterrows():
            X = list()
            for feature in feature_list:
                X.append(orjson.loads(row[feature]))
            y = orjson.loads(row["RT"])

            X = np.array(X)
            y = np.array(y)

            self.keys.append(row["SeqID"])
            self.ref_names.append(row["RefName"])
            self.Xs.append(X)
            self.ys.append(y)

        return self.keys, self.ref_names, self.Xs, self.ys

    def split(self, val_size:float, test_size:float, seed:int, batch_size:int, noise_scale:float):
        ref_names = dict()
        ref_names["Train"], ref_names["Test"] = train_test_split(self.dataset["RefName"].unique().tolist(), test_size=test_size, random_state=seed)
        ref_names["Train"], ref_names["Val"] = train_test_split(ref_names["Train"], test_size=val_size, random_state=seed)

        self.indices = dict()
        self.dataloaders = dict()

        for dataset_type in ["Train", "Val", "Test"]:
            self.indices[dataset_type] = np.isin(self.ref_names, ref_names[dataset_type])
            data = RNADataset(
                [value for value, index in zip(self.keys, self.indices[dataset_type]) if index],
                [value for value, index in zip(self.Xs, self.indices[dataset_type]) if index],
                [value for value, index in zip(self.ys, self.indices[dataset_type]) if index]
            )
            self.dataloaders[dataset_type] = utils.DataLoader(data, batch_size=1, shuffle=True)

        batch_data = RNADataset(
            [value for value, index in zip(self.keys, self.indices["Train"]) if index],
            [value for value, index in zip(self.Xs, self.indices["Train"]) if index],
            [value for value, index in zip(self.ys, self.indices["Train"]) if index]
        )
        if batch_size > 1:
            batch_sampler = LengthAwareSampler(batch_data, batch_size, noise_scale)
            self.dataloaders["Batch"] = utils.DataLoader(batch_data, batch_sampler=batch_sampler, collate_fn=collate_batch)
        else:
            self.dataloaders["Batch"] = self.dataloaders["Train"]

        return self.dataloaders
    
    def dump(self, data:dict[str,list], output_dir:str, with_original:bool=False):
        data = pd.DataFrame(data)
        if with_original:
            data = pd.merge(left=data, right=self.dataset, left_on="SeqID", right_on="SeqID", how="left")
        data.to_csv(f"{output_dir}/outputs.csv", index=False)
        print("outputs saved")

class Checkpoint:
    def __init__(self, checkpoint_pt:str=None, model_args:dict[str,]=None, optimizer_args:dict[str,]=None):
        self.buffer = io.BytesIO()
        if checkpoint_pt is not None:
            self.checkpoint = torch.load(checkpoint_pt, weights_only=False, map_location="cpu")
            self.checkpoint["Metadata"]["Parent"] = self.checkpoint["Metadata"]["Serial"]
            if optimizer_args is not None:
                self.checkpoint["OptimizerArgs"] = optimizer_args
        else:
            if model_args is None or optimizer_args is None:
                raise ValueError("model_args and optimizer_args are required")
            self.checkpoint = {
                "Model": None,
                "Optimizer": None,
                "ModelState": None,
                "OptimizerState": None,
                "ModelArgs": model_args,
                "OptimizerArgs": optimizer_args,
                "Metadata": {
                    "Parent": None,
                    "Serial": None,
                    "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Epoch": 0, # 1-based
                    "Evaluations": None,
                    "Losses": {
                        "Train": list(),
                        "Val": list()
                    }
                }
            }
    
    def load(self, module:str, model:str=None, optimizer:str=None, model_state:bool=False, optimizer_state:bool=False):
        module = importlib.import_module(module)

        if model is not None:
            self.checkpoint["Model"] = model
        model = getattr(module, self.checkpoint["Model"])(**self.checkpoint["ModelArgs"])
        if model_state:
            model.load_state_dict(self.checkpoint["ModelState"])
        model = model.to(dm.device)

        if optimizer is not None:
            self.checkpoint["Optimizer"] = optimizer
        optimizer = getattr(optim, self.checkpoint["Optimizer"])(model.parameters(), **self.checkpoint["OptimizerArgs"])
        if optimizer_state:
            optimizer.load_state_dict(self.checkpoint["OptimizerState"])
        
        return model, optimizer
    
    def update(self, losses:dict[str,float], delta_epochs:int):
        self.checkpoint["Metadata"]["Time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.checkpoint["Metadata"]["Losses"] = extend_dict(self.checkpoint["Metadata"]["Losses"], losses)
        self.checkpoint["Metadata"]["Epoch"] += delta_epochs

    def evaluate(self, evaluations:dict[str,float]):
        self.checkpoint["Metadata"]["Time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.checkpoint["Metadata"]["Evaluations"] = evaluations

    def checksum(self):
        serial = self.checkpoint["Metadata"]["Serial"]
        self.checkpoint["Metadata"]["Serial"] = None
        self.buffer.seek(0)
        self.buffer.truncate(0)
        torch.save(self.checkpoint, self.buffer)
        serialized = self.buffer.getvalue()
        md5sum = hashlib.md5(serialized).hexdigest()
        self.buffer.seek(0)
        self.buffer.truncate(0)
        self.checkpoint["Metadata"]["Serial"] = serial
        return md5sum

    def dump(self, model:nn.Module, optimizer:optim.Optimizer, output_dir:str):
        self.checkpoint["ModelState"] = model.state_dict()
        self.checkpoint["OptimizerState"] = optimizer.state_dict()
        serial = self.checksum()
        self.checkpoint["Metadata"]["Serial"] = serial
        torch.save(self.checkpoint, f"{output_dir}/checkpoint.pt")
        print("checkpoint saved")
        return serial

    def report(self, output_dir:str):
        with open(f"{output_dir}/report.json", "w") as file:
            json.dump(self.checkpoint["Metadata"], file, indent=4)
        print("report saved")

class Trainer:
    def __init__(self, dataset:Dataset, checkpoint:Checkpoint):
        print(f"using device: {dm.device}")
        self.dataset = dataset
        self.checkpoint = checkpoint

    def train(self, model:nn.Module, optimizer:optim.Optimizer, criterion:nn.Module, logits:bool):
        model.train()
        train_loss = 0.0
        for _, inputs, labels in tqdm(self.dataset.dataloaders["Batch"], desc=f"Training"):
            optimizer.zero_grad()
            outputs = model(inputs, logits)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _, inputs, labels in tqdm(self.dataset.dataloaders["Val"], desc=f"Validating"):
                outputs = model(inputs, logits)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        train_loss = train_loss / len(self.dataset.dataloaders["Batch"])
        val_loss = val_loss / len(self.dataset.dataloaders["Val"])

        return model, train_loss, val_loss
    
    def test(self, model:nn.Module, criterion:nn.Module, dataset_type:str):
        results = {"SeqID": list(), "Predictions": list(), "Score": list(), "Dataset": list()}
        model.eval()
        
        scores = list()
        with torch.no_grad():
            for keys, inputs, labels in tqdm(self.dataset.dataloaders[dataset_type], desc=f"Testing [{dataset_type} Set]"):
                outputs = model(inputs).cpu().numpy()
                labels = labels.cpu().numpy()
                for key, label, output in zip(keys, labels, outputs):
                    score = criterion(output, label)
                    scores.append(score)
                    results["SeqID"].append(key)
                    results["Predictions"].append(orjson.dumps(output.tolist()).decode())
                    results["Score"].append(score)
                    results["Dataset"].append(dataset_type)
        evaluation = np.mean(scores)
        return results, evaluation

    def autopilot(self, model:nn.Module, optimizer:optim.Optimizer, loss_fn:str, logits:bool, max_epochs:int, min_epochs:int, tolerance:float, output_dir:str):
        model = model.to(dm.device)
        criterion = getattr(nn, loss_fn)()

        total_epochs = self.checkpoint.checkpoint["Metadata"]["Epoch"] + max_epochs
        for _ in range(max_epochs):
            epoch = self.checkpoint.checkpoint["Metadata"]["Epoch"] + 1
            print(f"== Epoch {epoch}/{total_epochs} ==")
            model, train_loss, val_loss = self.train(model, optimizer, criterion, logits)
            print(f"training loss: {train_loss:.4f},", f"validation loss: {val_loss:.4f}")

            best_val_loss = min(self.checkpoint.checkpoint["Metadata"]["Losses"]["Val"], default=1.0)
            self.checkpoint.update({"Train": [train_loss], "Val": [val_loss]}, 1)

            if epoch >= min_epochs and val_loss < best_val_loss + tolerance:
                self.checkpoint.dump(model, optimizer, output_dir)
            self.checkpoint.report(output_dir)

        return model
        
    def evaluate(self, model:nn.Module, evaluation_fn:str, dataset_types:list[str], output_dir:str, report:bool=True):
        model = model.to(dm.device)
        criterion = getattr(metrics, evaluation_fn)

        outputs = {"SeqID": list(), "Predictions": list(), "Score": list(), "Dataset": list()}
        evaluations = dict()
    
        for dataset_type in dataset_types:
            output, evaluations[dataset_type] = self.test(model, criterion, dataset_type)
            outputs = extend_dict(outputs, output)

        self.dataset.dump(outputs, output_dir, with_original=False)
        self.checkpoint.evaluate(evaluations)
        
        if report:
            self.checkpoint.report(output_dir)
