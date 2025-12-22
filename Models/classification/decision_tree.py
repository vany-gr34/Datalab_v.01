from sklearn.tree import DecisionTreeClassifier
from Models.classification.BaseClassifier import BaseClassifier

class DecisionTreeClassifierLab(BaseClassifier):

    @staticmethod
    def hyperparameter_space():
        return {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }

    def build_model(self, **params):
        return DecisionTreeClassifier(**params)
