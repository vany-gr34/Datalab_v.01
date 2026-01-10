from itertools import product
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import joblib
import os


class BaseClusterer:

    def __init__(self):
        self.model = None
        self.best_params = None
        self.best_score = -np.inf

    def evaluate(self, X, labels):
        scores = {}

        # Some algorithms may assign a single cluster → silhouette undefined
        if len(set(labels)) > 1:
            scores["Silhouette"] = silhouette_score(X, labels)
            scores["DaviesBouldin"] = davies_bouldin_score(X, labels)
        else:
            scores["Silhouette"] = None
            scores["DaviesBouldin"] = None

        return scores

    def deploy(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def search_best_hyperparameters(self, X):
        param_grid = self.hyperparameter_space()
        keys, values = zip(*param_grid.items())

        best_score = -np.inf
        best_params = None

        for combination in product(*values):
            params = dict(zip(keys, combination))
            model = self.build_model(**params)

            labels = model.fit_predict(X)

            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
            else:
                score = -np.inf

            if score > best_score:
                best_score = score
                best_params = params

        self.best_params = best_params
        self.best_score = best_score
        return best_params

    def run(self, X, deploy=False, path=None):
        best_params = self.search_best_hyperparameters(X)

        self.model = self.build_model(**best_params)
        labels = self.model.fit_predict(X)

        metrics = self.evaluate(X, labels)

        if deploy and path:
            self.deploy(path)

        return labels, metrics
