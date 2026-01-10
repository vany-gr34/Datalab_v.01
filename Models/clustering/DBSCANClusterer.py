from sklearn.cluster import DBSCAN
from Models.clustering.BaseClusterer import BaseClusterer


class DBSCANClusterer(BaseClusterer):

    @staticmethod
    def hyperparameter_space():
        return {
            "eps": [0.3, 0.5, 0.7, 1.0],
            "min_samples": [3, 5, 10],
            "metric": ["euclidean", "manhattan"]
        }

    def build_model(self, **params):
        return DBSCAN(**params)

