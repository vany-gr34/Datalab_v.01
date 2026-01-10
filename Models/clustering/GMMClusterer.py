from sklearn.mixture import GaussianMixture
from Models.clustering.BaseClusterer import BaseClusterer


class GMMClusterer(BaseClusterer):

    @staticmethod
    def hyperparameter_space():
        return {
            "n_components": [2, 3, 4, 5],
            "covariance_type": ["full", "tied", "diag"],
            "max_iter": [100, 200]
        }

    def build_model(self, **params):
        return GaussianMixture(**params)
