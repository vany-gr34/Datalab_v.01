from sklearn.neighbors import KNeighborsClassifier
from Models.classification.BaseClassifier import BaseClassifier

class KNNClassifier(BaseClassifier):

    @staticmethod
    def hyperparameter_space():
        return {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
            "p": [1, 2]  # Manhattan vs Euclidean
        }

    def build_model(self, **params):
        return KNeighborsClassifier(**params)