from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# Base class for classification
class BaseClassifier:
    def __init__(self, model=None):
        self.model = model

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
