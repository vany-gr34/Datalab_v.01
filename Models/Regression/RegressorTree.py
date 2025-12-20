from sklearn.tree import DecisionTreeRegressor
from Models.Regression.BaseRegressor import BaseRegressor

class DecisionTreeRegressorLab(BaseRegressor):

    @staticmethod
    def hyperparameter_space():
        return {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10]
        }

    def build_model(self, **params):
        return DecisionTreeRegressor(random_state=42, **params)
