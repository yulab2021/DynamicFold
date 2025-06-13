import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_absolute_error
from tqdm import tqdm
import numpy as np
import datetime
import orjson
import json
import joblib

mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

def onehot_encode(sequence):
    tokens = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'U': [0, 0, 0, 1], 'N': [0.25, 0.25, 0.25, 0.25]}
    encoded = [tokens[nt] for nt in sequence]
    encoded = np.transpose(encoded).tolist()
    return encoded

def load_pipeline(pipeline_pkl):
    return joblib.load(pipeline_pkl)

class RNADataset:
    def __init__(self, dataset_csv:str, feature_list:list[str], test_size:float, seed:int):
        self.dataset = pd.read_csv(dataset_csv)
        self.load(feature_list)
        self.split(test_size, seed)

    def load(self, feature_list:list[str]):
        self.keys = list()
        self.ref_names = list()
        self.Xs = list()
        self.ys = list()

        for _, row in self.dataset.iterrows():
            X_entry = list()
            for feature in feature_list:
                X_entry.append(orjson.loads(row[feature]))
            y_entry = orjson.loads(row["RT"])

            X_entry = np.array(X_entry)
            y_entry = np.array(y_entry)

            self.keys.append(row["SeqID"])
            self.ref_names.append(row["RefName"])
            self.Xs.append(X_entry)
            self.ys.append(y_entry)
    
    def split(self, test_size:float, seed:int):
        train_ref_names, test_ref_names = train_test_split(self.dataset["RefName"].unique().tolist(), test_size=test_size, random_state=seed)

        train_indices = np.isin(self.ref_names, train_ref_names)
        test_indices = np.isin(self.ref_names, test_ref_names)

        self.train_keys = [value for value, index in zip(self.keys, train_indices) if index]
        self.X_train = [value for value, index in zip(self.Xs, train_indices) if index]
        self.y_train = [value for value, index in zip(self.ys, train_indices) if index]

        self.test_keys = [value for value, index in zip(self.keys, test_indices) if index]
        self.X_test = [value for value, index in zip(self.Xs, test_indices) if index]
        self.y_test = [value for value, index in zip(self.ys, test_indices) if index]

    def train_concat(self, bootstrap=0):
        if bootstrap > 0:
            indices = np.random.choice(len(self.train_keys), size=bootstrap, replace=True)
            self.train_keys = [self.train_keys[index] for index in indices]
            self.X_train = [self.X_train[index] for index in indices]
            self.y_train = [self.y_train[index] for index in indices]

        X_data = list()
        y_data = list()
        for X_entry, y_entry in zip(self.X_train, self.y_train):
            X_data.extend(X_entry.transpose().tolist())
            y_data.extend(y_entry.tolist())

        X_data = np.array(X_data)
        y_data = np.array(y_data)
        return X_data, y_data
    
    def test(self, pipeline, output_dir, params=dict()):
        evaluations = {"SeqID": list(), "Predictions": list(), "MAE": list(), "Dataset": list()}

        for seq_id, X_entry, y_entry in tqdm(zip(self.train_keys, self.X_train, self.y_train), desc="Testing [Train Set]", total=len(self.train_keys)):
            predictions = pipeline.predict(X_entry.transpose())
            MAE = mean_absolute_error(predictions, y_entry)
            evaluations["SeqID"].append(seq_id)
            evaluations["Predictions"].append(orjson.dumps(predictions.tolist()).decode())
            evaluations["MAE"].append(float(MAE))
            evaluations["Dataset"].append("Train")

        for seq_id, X_entry, y_entry in tqdm(zip(self.test_keys, self.X_test, self.y_test), desc="Testing [Test Set]", total=len(self.test_keys)):
            predictions = pipeline.predict(X_entry.transpose())
            MAE = mean_absolute_error(predictions, y_entry)
            evaluations["SeqID"].append(seq_id)
            evaluations["Predictions"].append(orjson.dumps(predictions.tolist()).decode())
            evaluations["MAE"].append(float(MAE))
            evaluations["Dataset"].append("Test")

        train_MMAE = np.mean([MAE for MAE, flag in zip(evaluations["MAE"], evaluations["Dataset"]) if flag == "Train"])
        test_MMAE = np.mean([MAE for MAE, flag in zip(evaluations["MAE"], evaluations["Dataset"]) if flag == "Test"])

        evaluations = pd.DataFrame(evaluations)
        evaluations.to_csv(f"{output_dir}/evaluations.csv", index=False)

        report = {
            "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "TrainMMAE": float(train_MMAE),
            "TestMMAE": float(test_MMAE)
        }
        report["Parameters"] = params
        with open(f"{output_dir}/report.json", "w") as report_file:
            json.dump(report, report_file, indent=4)

        joblib.dump(pipeline, f"{output_dir}/pipeline.pkl")
