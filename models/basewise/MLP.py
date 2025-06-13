from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import utils
import orjson
import sys

args_json = sys.argv[1]
args = orjson.loads(open(args_json, "r").read())

dataset = utils.RNADataset(args["DatasetCSV"], args["FeatureList"], args["TestSize"], args["Seed"])
pipeline = Pipeline([
    ('MLP', MLPRegressor(solver=args["Solver"], max_iter=args["MaxIter"]))
])
grid_search = GridSearchCV(estimator=pipeline, param_grid=args["ParameterGrid"], scoring=utils.mae_scorer, verbose=3, n_jobs=args["NumCores"])

grid_search.fit(*dataset.train_concat())
best_params = grid_search.best_params_
best_pipeline = grid_search.best_estimator_

dataset.test(best_pipeline, args["OutputDir"], best_params)