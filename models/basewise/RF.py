from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import utils
import orjson
import sys

args_json = sys.argv[1]
args = orjson.loads(open(args_json, "r").read())

dataset = utils.RNADataset(args["DatasetCSV"], args["FeatureList"], args["TestSize"], args["Scorer"], args["Seed"])
pipeline = Pipeline([
    ('RF', RandomForestRegressor())
])
grid_search = GridSearchCV(estimator=pipeline, param_grid=args["ParameterGrid"], scoring=dataset.scorer, verbose=3, n_jobs=args["NumCores"])

grid_search.fit(*dataset.train_concat(args["Bootstrap"]))
best_params = grid_search.best_params_
best_pipeline = grid_search.best_estimator_

dataset.test(best_pipeline, args["Scorer"], args["OutputDir"], best_params)