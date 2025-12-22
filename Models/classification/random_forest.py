from sklearn.ensemble import RandomForestClassifier
from Models.classification.BaseClassifier import BaseClassifier

class RandomForestClassifierModel(BaseClassifier):

    @staticmethod
    def hyperparameter_space():
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }

    def build_model(self, **params):
        return RandomForestClassifier(**params)
