import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

## Hyperparameter tuning config    
param_grids = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
    #    "max_depth": [None, 10, 20, 30],
    #    "min_samples_split": [2, 5, 10],
    #    "min_samples_leaf": [1, 2, 4],
    #    "max_features": ["auto", "sqrt", "log2"],
    #    "bootstrap": [True, False]
    },

    "Decision Tree": {
        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
        "splitter": ["best", "random"]
    #    "max_depth": [None, 10, 20, 30],
    #    "min_samples_split": [2, 5],
    #    "min_samples_leaf": [1, 2],
    #    "max_features": ["auto", "sqrt", "log2", None]
    },

    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0]
    #    "max_depth": [3, 5, 7],
    #    "min_samples_split": [2, 5],
    #    "min_samples_leaf": [1, 2],
    #    "max_features": ["auto", "sqrt", "log2"]
    },

    "Linear Regression": {
    #    "fit_intercept": [True, False],
    #    "positive": [True, False]
    },

    "Ridge": {
        "alpha": [0.1, 1.0, 10.0, 100.0],
    #    "solver": ["auto", "svd", "cholesky", "lsqr", "saga"],
    #    "fit_intercept": [True, False]
    },

    "Lasso": {
        "alpha": [0.001, 0.01, 0.1, 1.0],
    #    "max_iter": [1000, 5000],
    #    "fit_intercept": [True, False],
    #    "selection": ["cyclic", "random"]
    },

    "ElasticNet": {
        "alpha": [0.01, 0.1, 1.0],
        "l1_ratio": [0.1, 0.5, 0.9],
    #    "max_iter": [1000, 5000],
    #    "fit_intercept": [True, False],
    #    "selection": ["cyclic", "random"]
    },

    "HuberRegressor": {
        "epsilon": [1.1, 1.35, 1.5, 2.0],
        "alpha": [0.0001, 0.001, 0.01],
    #    "fit_intercept": [True, False],
    #    "max_iter": [100, 500]
    },

#    "ExtraTrees": {
#        "n_estimators": [100, 200],
    #    "max_depth": [None, 10, 20, 30],
    #    "min_samples_split": [2, 5],
    #    "min_samples_leaf": [1, 2],
    #    "max_features": ["auto", "sqrt", "log2"],
    #    "bootstrap": [True, False]
    #},

    "SVR": {
        "C": [0.1, 1.0, 10.0],
        "epsilon": [0.01, 0.1, 1.0],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"]
    },

    "K-Neighbors Regressor": {
        "n_neighbors": [3, 5, 7, 9],
    #    "weights": ["uniform", "distance"],
    #    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    #    "p": [1, 2]
    },

    "XGB Regressor": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
    #    "max_depth": [3, 5, 7],
    #    "subsample": [0.8, 1.0],
    #    "colsample_bytree": [0.8, 1.0],
    #    "reg_alpha": [0, 0.1, 1.0],
    #    "reg_lambda": [1.0, 1.5, 2.0],
    #    "gamma": [0, 0.1, 0.3],
    #    "booster": ["gbtree", "dart"]
    },

    "CatBoosting Regressor": {
        "iterations": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [3, 5, 7, 10],
    #    "l2_leaf_reg": [1, 3, 5],
    #    "bootstrap_type": ["Bayesian", "Bernoulli", "MVS"]
    },

    "AdaBoost Regressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1.0],
    #    "loss": ["linear", "square", "exponential"]
    },

    "GaussianProcess": {
        "alpha": [1e-10, 1e-5, 1e-2],
    #    "normalize_y": [True, False]
    },

#    "Dummy": {
#        "strategy": ["mean", "median", "quantile"],
    #    "quantile": [0.1, 0.5, 0.9]
#    }
}


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train,y_train,X_test,y_test,models, param=param_grids):
    '''
    This function is to train the model and predict using test data.
    It will be done against all the passed models (in a dict) and their 
    respective r2 score will be stored in dict and return.
    '''
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param_tun = param[list(models.keys())[i]]


            grid = GridSearchCV(model, param_tun, scoring="neg_root_mean_squared_error", cv=3)
            grid.fit(X_train, y_train)

            model.set_params(**grid.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)