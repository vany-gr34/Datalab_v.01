from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from preprocessing.BaseTransformer import BaseTransformer
import pandas as pd

class CategoricalEncoder(BaseTransformer):
    def __init__(self, method="onehot", handle_unknown="ignore"):
        self.method = method
        self.handle_unknown = handle_unknown
        self.encoder = None
        self.single_column = False  # for LabelEncoder

    def fit(self, X):
        # Ensure X is a DataFrame
        if isinstance(X, pd.Series):
            X = X.to_frame()
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.method == "onehot":
            self.encoder = OneHotEncoder(handle_unknown=self.handle_unknown, sparse=False)
            self.encoder.fit(X)
        elif self.method == "ordinal":
            self.encoder = OrdinalEncoder(handle_unknown=self.handle_unknown)
            self.encoder.fit(X)
        elif self.method == "label":
            if X.shape[1] != 1:
                raise ValueError("LabelEncoder can only be applied to a single column")
            self.encoder = LabelEncoder()
            self.encoder.fit(X.iloc[:,0])
            self.single_column = True
        else:
            raise ValueError(f"Unknown encoding method: {self.method}")

    def transform(self, X):
        if isinstance(X, pd.Series):
            X = X.to_frame()
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.method in ["onehot", "ordinal"]:
            return self.encoder.transform(X)
        elif self.method == "label":
            return self.encoder.transform(X.iloc[:,0])
