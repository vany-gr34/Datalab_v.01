from xgboost import XGBRegressor
from Models.Regression.BaseRegressor import BaseRegressor

class XGBoostRegressor(BaseRegressor):

    @staticmethod
    def hyperparameter_space():
        return {
            "n_estimators": [100, 200],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0]
        }

    def build_model(self, **params):
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            **params
        )
