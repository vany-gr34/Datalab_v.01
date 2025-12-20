from sklearn.linear_model import LinearRegression
from Models.Regression.BaseRegressor import BaseRegressor

class LinearRegressor(BaseRegressor):

    @staticmethod
    def hyperparameter_space():
        return {
            "fit_intercept": [True, False],
            "positive": [True, False]
        }

    def build_model(self, **params):
        return LinearRegression(**params)
