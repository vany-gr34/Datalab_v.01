from sklearn.svm import SVR
from Models.Regression.BaseRegressor import BaseRegressor

class SVRRegressor(BaseRegressor):

    @staticmethod
    def hyperparameter_space():
        return {
            "C": [0.1, 1, 10],
            "epsilon": [0.01, 0.1],
            "kernel": ["rbf", "linear"]
        }

    def build_model(self, **params):
        return SVR(**params)
