from sklearn.linear_model import LogisticRegression
from Models.classification.BaseClassifier import BaseClassifier

# Logistic Regression
class LogisticRegressionClassifier(BaseClassifier):

    @staticmethod
    def hyperparameter_space():
        return {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs", "saga"],
            "max_iter": [100, 200]
        }

    def build_model(self, **params):
        return LogisticRegression(**params)