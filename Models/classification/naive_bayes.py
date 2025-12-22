from sklearn.naive_bayes import GaussianNB
from Models.classification.BaseClassifier import BaseClassifier

class NaiveBayesClassifier(BaseClassifier):

    @staticmethod
    def hyperparameter_space():
        # GaussianNB has no major hyperparameters, can keep empty
        return {}

    def build_model(self, **params):
        return GaussianNB(**params)
