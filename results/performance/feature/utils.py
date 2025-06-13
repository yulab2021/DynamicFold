from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
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
    
    def get_device(self):
        return self.device
    
    def set_device(self, device:str):
        self.device = torch.device(device)

dm = DeviceManager()

def onehot_encode(sequence:str):
    tokens = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'U': [0, 0, 0, 1], 'N': [0.25, 0.25, 0.25, 0.25]}
    encoded = [tokens[nt] for nt in sequence]
    encoded = np.transpose(encoded).tolist()
    return encoded

def extend_dict(x:dict[str,list], y:dict[str,list]):
    for key, value in y.items():
        if (not isinstance(key, str)) or (not isinstance(value, list)):
            raise ValueError("invalid dict format")
        if key in x:
            x[key].extend(value)
        else:
            x[key] = value
    return x

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
    max_len = max([X.shape[1] for X in Xs])
    Xs = [F.pad(X, (0, max_len - X.shape[1]), "constant", 0) for X in Xs]
    ys = [F.pad(y, (0, max_len - y.shape[0]), "constant", 0) for y in ys]
    Xs = torch.stack(Xs)
    ys = torch.stack(ys)
    return keys, Xs, ys

class RNADataset(utils.Dataset):
    def __init__(self, keys, X:list[np.ndarray], y:list[np.ndarray]):
        assert len(keys) == len(X) == len(y)
        self.keys = keys
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.keys)
    
    def entry_lengths(self):
        return [y.shape[0] for y in self.y]

    def __getitem__(self, index:int):
        key = self.keys[index]
        X_entry = torch.tensor(self.X[index], dtype=torch.float32).to(dm.device)
        y_entry = torch.tensor(self.y[index], dtype=torch.float32).to(dm.device)
        return key, X_entry, y_entry

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
            X_entry = list()
            for feature in feature_list:
                if feature == "SQ":
                    X_entry.extend(onehot_encode(row["SQ"]))
                else:
                    X_entry.append(orjson.loads(row[feature]))
            y_entry = orjson.loads(row["RT"])

            X_entry = np.array(X_entry)
            y_entry = np.array(y_entry)

            self.keys.append(row["SeqID"])
            self.ref_names.append(row["RefName"])
            self.Xs.append(X_entry)
            self.ys.append(y_entry)

        return self.keys, self.ref_names, self.Xs, self.ys

    def split(self, val_size:float, test_size:float, seed:int, batch_size:int, noise_scale:float):
        train_ref_names, test_ref_names = train_test_split(self.dataset["RefName"].unique().tolist(), test_size=test_size, random_state=seed)
        train_ref_names, val_ref_names = train_test_split(train_ref_names, test_size=val_size, random_state=seed)

        self.train_indices = np.isin(self.ref_names, train_ref_names)
        self.val_indices = np.isin(self.ref_names, val_ref_names)
        self.test_indices = np.isin(self.ref_names, test_ref_names)

        train_data = RNADataset(
            [value for value, index in zip(self.keys, self.train_indices) if index],
            [value for value, index in zip(self.Xs, self.train_indices) if index],
            [value for value, index in zip(self.ys, self.train_indices) if index]
        )
        val_data = RNADataset(
            [value for value, index in zip(self.keys, self.val_indices) if index],
            [value for value, index in zip(self.Xs, self.val_indices) if index],
            [value for value, index in zip(self.ys, self.val_indices) if index]
        )
        test_data = RNADataset(
            [value for value, index in zip(self.keys, self.test_indices) if index],
            [value for value, index in zip(self.Xs, self.test_indices) if index],
            [value for value, index in zip(self.ys, self.test_indices) if index]
        )

        if batch_size > 1:
            train_sampler = LengthAwareSampler(train_data, batch_size, noise_scale)
            self.batch_dataloader = utils.DataLoader(train_data, batch_sampler=train_sampler, collate_fn=collate_batch)
        else:
            self.batch_dataloader = utils.DataLoader(train_data, batch_size=1, shuffle=True)
            
        self.train_dataloader = utils.DataLoader(train_data, batch_size=1, shuffle=True)
        self.val_dataloader = utils.DataLoader(val_data, batch_size=1, shuffle=True)
        self.test_dataloader = utils.DataLoader(test_data, batch_size=1, shuffle=True)

        return self.train_dataloader, self.val_dataloader, self.test_dataloader
    
    def dump(self, data:dict[str,list], output_dir:str, with_original:bool=False):
        data = pd.DataFrame(data)
        if with_original:
            data = pd.merge(left=data, right=self.dataset, left_on="SeqID", right_on="SeqID", how="left")
        data.to_csv(f"{output_dir}/evaluations.csv", index=False)
        print("evaluations saved")

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
                    "Evaluations": {
                        "TrainMMAE": None,
                        "ValMMAE": None,
                        "TestMMAE": None
                    },
                    "Losses": {
                        "TrainLoss": list(),
                        "ValLoss": list()
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
    
    def update(self, losses:dict[str,float]):
        self.checkpoint["Metadata"]["Time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.checkpoint["Metadata"]["Epoch"] += 1
        self.checkpoint["Metadata"]["Losses"]["TrainLoss"].append(losses["TrainLoss"])
        self.checkpoint["Metadata"]["Losses"]["ValLoss"].append(losses["ValLoss"])

    def evaluate(self, evaluations:dict[str,float]):
        self.checkpoint["Metadata"]["Time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.checkpoint["Metadata"]["Evaluations"]["TrainMMAE"] = evaluations["TrainMMAE"]
        self.checkpoint["Metadata"]["Evaluations"]["ValMMAE"] = evaluations["ValMMAE"]
        self.checkpoint["Metadata"]["Evaluations"]["TestMMAE"] = evaluations["TestMMAE"]

    def checksum(self):
        self.checkpoint["Metadata"]["Serial"] = None
        self.buffer.seek(0)
        self.buffer.truncate(0)
        torch.save(self.checkpoint, self.buffer)
        serialized = self.buffer.getvalue()
        md5sum = hashlib.md5(serialized).hexdigest()
        self.buffer.seek(0)
        self.buffer.truncate(0)
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
        for _, inputs, labels in tqdm(self.dataset.batch_dataloader, desc=f"Training"):
            optimizer.zero_grad()
            outputs = model(inputs, logits)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _, inputs, labels in tqdm(self.dataset.val_dataloader, desc=f"Validating"):
                outputs = model(inputs, logits)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        train_loss = train_loss / len(self.dataset.batch_dataloader)
        val_loss = val_loss / len(self.dataset.val_dataloader)

        return model, train_loss, val_loss
    
    def test(self, model:nn.Module, dataloader:utils.DataLoader, dataset_type:str):
        results = {"SeqID": list(), "Predictions": list(), "MAE": list(), "Dataset": list()}
        model.eval()
        
        MAEs = list()
        with torch.no_grad():
            for seq_ids, X_entries, y_entries in tqdm(dataloader, desc=f"Testing [{dataset_type} Set]"):
                predictions = model(X_entries).cpu().numpy()
                for seq_id, y_entry, prediction in zip(seq_ids, y_entries, predictions):
                    MAE = mean_absolute_error(prediction, y_entry.cpu().numpy())
                    MAEs.append(MAE)
                    results["SeqID"].append(seq_id)
                    results["Predictions"].append(orjson.dumps(prediction.tolist()).decode())
                    results["MAE"].append(MAE)
                    results["Dataset"].append(dataset_type)
        MMAE = np.mean(MAEs)
        return results, MMAE

    def autopilot(self, model:nn.Module, optimizer:optim.Optimizer, loss_fn:str, logits:bool, max_epochs:int, min_epochs:int, tolerance:float, output_dir:str):
        model = model.to(dm.device)
        criterion = getattr(nn, loss_fn)()

        for epoch in range(max_epochs):
            print(f"== Epoch {epoch+1}/{max_epochs} ==")
            model, train_loss, val_loss = self.train(model, optimizer, criterion, logits)
            print(f"training loss: {train_loss:.4f},", f"validation loss: {val_loss:.4f}")

            best_val_loss = min(self.checkpoint.checkpoint["Metadata"]["Losses"]["ValLoss"], default=1.0)
            self.checkpoint.update({"TrainLoss": train_loss, "ValLoss": val_loss})
            if epoch >= min_epochs and val_loss < best_val_loss + tolerance:
                self.evaluate(model, output_dir, report=False)
                self.checkpoint.dump(model, optimizer, output_dir)
            
            self.checkpoint.report(output_dir)

        return model
        
    def evaluate(self, model:nn.Module, output_dir:str, report:bool=True):
        model = model.to(dm.device)
    
        train_evaluations, train_MMAE = self.test(model, self.dataset.train_dataloader, "Train")
        val_evaluations, val_MMAE = self.test(model, self.dataset.val_dataloader, "Val")
        test_evaluations, test_MMAE = self.test(model, self.dataset.test_dataloader, "Test")
        print(f"training MMAE: {train_MMAE:.4f},", f"validation MMAE: {val_MMAE:.4f},", f"testing MMAE: {test_MMAE:.4f}")

        evaluations = {"SeqID": list(), "Predictions": list(), "MAE": list(), "Dataset": list()}
        evaluations = extend_dict(evaluations, train_evaluations)
        evaluations = extend_dict(evaluations, val_evaluations)
        evaluations = extend_dict(evaluations, test_evaluations)

        self.dataset.dump(evaluations, output_dir, with_original=False)
        self.checkpoint.evaluate({"TrainMMAE": train_MMAE, "ValMMAE": val_MMAE, "TestMMAE": test_MMAE})
        
        if report:
            self.checkpoint.report(output_dir)
