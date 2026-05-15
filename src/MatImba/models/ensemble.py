import os
import sys
import joblib
import pandas as pd
from sklearn.utils import indexable
from sklearn.pipeline import make_pipeline
from ..dataset.imba import focus_func, get_weights, estimate_density
from ..utils.losses import calc_sera
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor

import sklearn
sklearn.set_config(enable_metadata_routing=True)

def get_obj(obj_name):
    """
    Return object from the object name, need to import the obj
    """
    try:
        return getattr(sys.modules[__name__], obj_name)
    except AttributeError:
        print(f"Please import the {obj_name} first.")


class gbr_ensemble():
    def __init__(
        self, gbr_params = None, n_gbrs = 20, lds = False,
        scaler = "MaxAbsScaler", test_size = 0.1, n_jobs = 5,
        random_seeds = {
            "train_test": 19, "bagging": 119
        }, outdir = None
    ):
        self.lds = lds
        self.random_seeds = random_seeds
        self.test_size = test_size
        # self._load_data(*data_files, target_name = target_name)
        self.n_gbrs = n_gbrs
        self.scaler = get_obj(scaler)
        self.n_jobs = n_jobs
        self.outdir = os.getcwd() if outdir is None else outdir

        self.gbr_params = {
            'n_estimators'  : 1500,
            'learning_rate' : 0.005,
            'max_depth'     : 6,
            'loss'          : 'huber',

            'subsample'     : 0.65,
            'alpha'         : 0.8,
        } if gbr_params is None else gbr_params
        
        self.gbr = make_pipeline(
            self.scaler(), GradientBoostingRegressor(**self.gbr_params).set_fit_request(sample_weight=True)
        )
        
        self.bagged_gbrs = BaggingRegressor(
            estimator = self.gbr,
            n_estimators = self.n_gbrs, n_jobs = self.n_jobs,
            random_state = self.random_seeds["bagging"]
        )
        self.final_models = None
        
    def load_data(self, *data_files, target_name, focus = None, **lds_params):
        file_num = len(data_files)
        if file_num == 0:
            print("No data files specified!")
            self.train_x = None
            self.train_y = None
            self.train_weights = None
        else:
            self.data = data_files
            if file_num == 1:
                data_df = pd.read_csv(self.data[0])
                X = data_df.iloc[:, :-1]
                Y = data_df.iloc[:, -1]
            elif file_num == 2:
                X = pd.read_csv(self.data[0])
                Y = pd.read_csv(self.data[1])[[target_name]]
                not_na_inds = Y[target_name].notna()
                X = X[not_na_inds]
                Y = Y[not_na_inds][target_name].values
            else:
                raise ValueError("Too many datafiles to parse.")

            label_densities = estimate_density(Y, smooth = "convolve")
            label_relevances = get_weights(label_densities, eps = 0)

            if self.lds:
                if focus is not None:
                    focus_den = focus_func(Y, **focus)
                    label_densities = (label_densities - focus_den).clip(1e-4, 1)
                label_weights = get_weights(label_densities, eps = 0.05, **lds_params)
                self.train_x, self.test_x, self.train_y, self.test_y, self.train_weights, self.test_weights, self.train_relevances, self.test_relevances, self.train_densities, self.test_densities = train_test_split(
                    X, Y, label_weights, label_relevances, label_densities, test_size = self.test_size, random_state = self.random_seeds["train_test"]
                )
            else:
                self.train_x, self.test_x, self.train_y, self.test_y, self.train_relevances, self.test_relevances, self.train_densities, self.test_densities = train_test_split(
                    X, Y, label_relevances, label_densities, test_size = self.test_size, random_state = self.random_seeds["train_test"]
                )
                self.train_weights = None
                self.test_weights = None

    def load(self, saved_models):
        self.final_models = joblib.load(saved_models)
        
    def save(self, filename, prefix = "contorl"):
        if self.final_models is not None:
            joblib.dump(self.final_models, os.path.join(self.outdir, filename))
            for i, model in enumerate(self.final_models.estimators_):
                joblib.dump(model, os.path.join(self.outdir, f"{prefix}_{i}.pkl"))
        else:
            joblib.dump(self.bagged_gbrs, os.path.join(self.outdir, filename))
            for i, model in enumerate(self.bagged_gbrs.estimators_):
                joblib.dump(model, os.path.join(self.outdir, f"{prefix}_{i}.pkl"))
        
    def fit(self, *train_data):
        num_inputs = len(train_data)
        if num_inputs == 0:
            self.final_models = self.bagged_gbrs.fit(self.train_x, self.train_y, sample_weight = self.train_weights)
        else:
            if num_inputs == 1:
                X = train_data[0][:, :-1]
                Y = train_data[0][:, -1]
                weights = None
            elif num_inputs == 2:
                X = train_data[0]
                Y = train_data[1]
                weights = None
            elif num_inputs == 3:
                X = train_data[0]
                Y = train_data[1]
                weights = train_data[2]
            self.final_models = self.bagged_gbrs.fit(X, Y, sample_weight = weights)
        
    def predict(self, inputs = None, targets = None):
        if inputs is None:
            inputs = self.test_x
        if self.final_models is not None:
            outputs = self.final_models.predict(inputs)
        else:
            outputs = self.bagged_gbrs.predict(inputs)
        return outputs