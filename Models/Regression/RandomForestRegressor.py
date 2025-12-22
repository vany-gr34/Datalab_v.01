from sklearn.ensemble import RandomForestRegressor
from Models.Regression.BaseRegressor import BaseRegressor

class RandomForestRegressorLab(BaseRegressor):

    @staticmethod
    def hyperparameter_space():
        return {
            "n_estimators": [50, 100],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }

    def build_model(self, **params):
        return RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **params
        )
