from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib
import os


class BaseRegressor:

    def __init__(self):
        self.model = None
        self.best_params = None
        self.best_score = -np.inf

    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

    def deploy(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

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
            score = r2_score(y_valid, y_pred)

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
        metrics = self.evaluate(y_valid, y_pred)

        if deploy and path:
            self.deploy(path)

        return y_pred, metrics
