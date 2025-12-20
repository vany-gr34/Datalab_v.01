from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from Models.Regression.BaseRegressor import BaseRegressor

class PolynomialRegressor(BaseRegressor):

    @staticmethod
    def hyperparameter_space():
        return {
            "degree": [2, 3, 4],
            "fit_intercept": [True, False]
        }

    def build_model(self, degree, fit_intercept):
        return Pipeline([
            ("poly", PolynomialFeatures(degree=degree)),
            ("lr", LinearRegression(fit_intercept=fit_intercept))
        ])
