from sklearn.linear_model import Ridge
from Models.Regression.BaseRegressor import BaseRegressor
class RidgeRegressor(BaseRegressor):

    @staticmethod
    def hyperparameter_space():
        return {
            "alpha": [0.01, 0.1, 1.0, 10.0],
            "fit_intercept": [True, False]
        }

    def build_model(self, **params):
        return Ridge(**params)
