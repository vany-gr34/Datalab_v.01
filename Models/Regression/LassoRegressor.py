from sklearn.linear_model import Lasso
from Models.Regression.BaseRegressor import BaseRegressor

class LassoRegressor(BaseRegressor):

    @staticmethod
    def hyperparameter_space():
        return {
            "alpha": [0.001, 0.01, 0.1, 1.0],
            "max_iter": [1000, 5000]
        }

    def build_model(self, **params):
        return Lasso(**params)
