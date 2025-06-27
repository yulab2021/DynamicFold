import utils
import argparse
import orjson

parser = argparse.ArgumentParser()
parser.add_argument("--configs", "-c", type=str, required=True, help="Path to the JSON configuration file.")
parser.add_argument("--device", "-d", type=str, required=False, help="Device to use for training: 'cpu', 'cuda', or 'mps'.")
args = parser.parse_args()

if args.device is not None:
    utils.dm.set(args.device)

configs:dict = orjson.loads(open(args.configs, "r").read())
dataset = utils.Dataset(**configs["DatasetArgs"])
checkpoint = utils.Checkpoint(checkpoint_pt=configs.get("CheckpointPT"), model_args=configs.get("ModelArgs"), optimizer_args=configs.get("OptimizerArgs"))
trainer = utils.Trainer(dataset, checkpoint)

if configs["Mode"] == "New":
    model, optimizer = checkpoint.load(configs["Module"], configs.get("Model"), configs.get("Optimizer"), model_state=False, optimizer_state=False)
    model = trainer.autopilot(model, optimizer, **configs["AutopilotArgs"])
elif configs["Mode"] == "Resume":
    model, optimizer = checkpoint.load(configs["Module"], configs.get("Model"), configs.get("Optimizer"), model_state=True, optimizer_state=True)
    model = trainer.autopilot(model, optimizer, **configs["AutopilotArgs"])
elif configs["Mode"] == "Transfer":
    model, optimizer = checkpoint.load(configs["Module"], configs.get("Model"), configs.get("Optimizer"), model_state=True, optimizer_state=False)
    model = trainer.autopilot(model, optimizer, **configs["AutopilotArgs"])
elif configs["Mode"] == "Evaluate":
    model, _ = checkpoint.load(configs["Module"], configs.get("Model"), configs.get("Optimizer"), model_state=True, optimizer_state=False)
    model = trainer.evaluate(model, **configs["EvaluateArgs"])
else:
    raise ValueError(f"invalid mode: {configs["Mode"]}")
