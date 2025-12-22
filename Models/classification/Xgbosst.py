from xgboost import XGBClassifier
from Models.classification.BaseClassifier import BaseClassifier

class XGBoostClassifier(BaseClassifier):

    @staticmethod
    def hyperparameter_space():
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0]
        }

    def build_model(self, **params):
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss', **params)
