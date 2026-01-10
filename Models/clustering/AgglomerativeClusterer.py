from sklearn.cluster import AgglomerativeClustering
from Models.clustering.BaseClusterer import BaseClusterer


class AgglomerativeClusterer(BaseClusterer):

    @staticmethod
    def hyperparameter_space():
        return {
            "n_clusters": [2, 3, 4, 5, 8],
            "linkage": ["ward", "complete", "average"],
            "metric": ["euclidean"]
        }

    def build_model(self, **params):
        return AgglomerativeClustering(**params)
