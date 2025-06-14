import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.dummy import DummyRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models, param_grids


@dataclass
class  ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "HuberRegressor": HuberRegressor(),
            #    "ExtraTrees": ExtraTreesRegressor(),
                "SVR": SVR(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "GaussianProcess": GaussianProcessRegressor(),
            #    "Dummy": DummyRegressor(),
            }

            model_report:dict=evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=param_grids
            )

            ## Assumption: unique key with maximum value
            #best_model_name = max(model_report, key=model_report.get)
            #best_model_score = model_report[best_model_name]
            

            ## For multiple key with maximum value
            max_value = max(model_report.values())
            max_keys = [k for k, v in model_report.items() if v == max_value]
            best_model=models[max_keys[0]]

            

            #if best_model_score<0.6:
            if max_value<0.6:
                raise CustomException("No Best Model Found")
            logging.info(f"Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square=r2_score(y_test,predicted)

            #return r2_square,best_model_name, best_model_score
            return max_keys,max_value 
            #return best_model_name,best_model_score
            #return model_report
        


        except Exception as e:
            raise CustomException(e,sys)
