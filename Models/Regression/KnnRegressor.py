from sklearn.neighbors import KNeighborsRegressor

from Models.Regression.BaseRegressor import BaseRegressor
class KNNRegressor(BaseRegressor):

    @staticmethod
    def hyperparameter_space():
        return {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
            "p": [1, 2]  # Manhattan vs Euclidean
        }

    def build_model(self, **params):
        return KNeighborsRegressor(**params)
