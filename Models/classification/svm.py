from sklearn.svm import SVC
from Models.classification.BaseClassifier import BaseClassifier

class SVCClassifier(BaseClassifier):

    @staticmethod
    def hyperparameter_space():
        return {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"]
        }

    def build_model(self, **params):
        return SVC(probability=True, **params)
