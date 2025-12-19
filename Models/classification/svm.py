from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


class SVMModel:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_valid):
        return self.model.predict(X_valid)

    def evaluate(self, y_valid, y_pred):
        return {
            "accuracy": accuracy_score(y_valid, y_pred),
            "precision": precision_score(y_valid, y_pred, average="weighted"),
            "recall": recall_score(y_valid, y_pred, average="weighted"),
            "f1": f1_score(y_valid, y_pred, average="weighted"),
        }

    def deploy(self, path="deployment/Classification/svm.pkl"):
        joblib.dump(self.model, path)

    def hyperparameters(self):
        return {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        }

    def run(self, X_train, X_valid, y_train, y_valid, **hyperparams):
        self.model = SVC(**hyperparams)
        self.train(X_train, y_train)
        y_pred = self.predict(X_valid)
        metrics = self.evaluate(y_valid, y_pred)
        self.deploy()
        return y_pred, metrics
