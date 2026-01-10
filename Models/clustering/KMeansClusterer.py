from sklearn.cluster import KMeans
from Models.clustering.BaseClusterer import BaseClusterer


class KMeansClusterer(BaseClusterer):

    @staticmethod
    def hyperparameter_space():
        return {
            "n_clusters": [2, 3, 4, 5, 8],
            "init": ["k-means++", "random"],
            "n_init": [10, 20],
            "max_iter": [300, 500]
        }

    def build_model(self, **params):
        return KMeans(**params)
