

from Models.classification.logistic_regression import LogisticRegressionClassifier
from Models.classification.knn import KNNClassifier
from Models.classification.random_forest import RandomForestClassifierModel as RandomForestClassifier
from Models.classification.decision_tree import DecisionTreeClassifierLab
from Models.classification.svm import SVCClassifier
from Models.classification.naive_bayes import NaiveBayesClassifier
#from Models.classification.neural_network import NeuralNetworkClassifier
from Models.classification.Xgbosst import XGBoostClassifier


from Models.Regression.LinearRegressor import LinearRegressor
from Models.Regression.RidgeRegressor import RidgeRegressor
from Models.Regression.LassoRegressor import LassoRegressor
from Models.Regression.PolyRegressor import PolynomialRegressor
from Models.Regression.KnnRegressor import KNNRegressor
from Models.Regression.SVr import SVRRegressor
from Models.Regression.RegressorTree import DecisionTreeRegressorLab
from Models.Regression.RandomForestRegressor import RandomForestRegressorLab
from Models.Regression.XgboostRegressor import XGBoostRegressor


CLASSIFIERS = {
    "Logistic Regression": LogisticRegressionClassifier,
    "KNN": KNNClassifier,
    "Random Forest": RandomForestClassifier,
    "Decision Tree": DecisionTreeClassifierLab,
    "SVM": SVCClassifier,
    "Naive Bayes": NaiveBayesClassifier,
    #"Neural Network": NeuralNetworkClassifier,
    "XGBoost": XGBoostClassifier,
}




REGRESSORS = {
    "Linear Regression": LinearRegressor,
    "Ridge Regression": RidgeRegressor,
    "Lasso Regression": LassoRegressor,
    "Polynomial Regression": PolynomialRegressor,
    "KNN Regressor": KNNRegressor,
    "SVR": SVRRegressor,
    "Decision Tree Regressor": DecisionTreeRegressorLab,
    "Random Forest Regressor": RandomForestRegressorLab,
    "XGBoost Regressor": XGBoostRegressor,
}
 