from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from itertools import product
import numpy as np

# Base class for classification
class BaseClassifier:
    def __init__(self, model=None):
        self.model = model
        self.best_params = None
        self.best_score = -np.inf

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # Optional: only works if the classifier supports predict_proba
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("This model does not support predict_proba")

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average="weighted"),
            "recall": recall_score(y, y_pred, average="weighted"),
            "f1": f1_score(y, y_pred, average="weighted"),
            "confusion_matrix": confusion_matrix(y, y_pred)
        }
        return metrics

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

    def search_best_hyperparameters(self, X_train, X_valid, y_train, y_valid):
        param_grid = self.hyperparameter_space()
        keys, values = zip(*param_grid.items())

        best_score = -np.inf
        best_params = None

        for combination in product(*values):
            params = dict(zip(keys, combination))
            model = self.build_model(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_valid)
            score = accuracy_score(y_valid, y_pred)

            if score > best_score:
                best_score = score
                best_params = params

        self.best_params = best_params
        self.best_score = best_score
        return best_params

    def run(self, X_train, X_valid, y_train, y_valid, deploy=False, path=None):
        best_params = self.search_best_hyperparameters(
            X_train, X_valid, y_train, y_valid
        )

        self.model = self.build_model(**best_params)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_valid)
        metrics = self.evaluate(X_valid, y_valid)

        if deploy and path:
            self.save(path)

        return y_pred, metrics
