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

        # Convert to string to handle mixed types and NAType
        X = X.astype(str)

        if self.method == "onehot":
            self.encoder = OneHotEncoder(handle_unknown=self.handle_unknown, sparse=False)
            self.encoder.fit(X)
        elif self.method == "ordinal":
            # OrdinalEncoder does not support 'ignore'; use 'error' or 'use_encoded_value'
            ordinal_handle_unknown = 'error' if self.handle_unknown == 'ignore' else self.handle_unknown
            self.encoder = OrdinalEncoder(handle_unknown=ordinal_handle_unknown)
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
        # Return a DataFrame when possible so callers can reassign columns
        if self.method == "onehot":
            arr = self.encoder.transform(X)
            try:
                cols = self.encoder.get_feature_names_out(X.columns)
            except Exception:
                cols = [f"{c}_{i}" for i, c in enumerate(X.columns)]
            return pd.DataFrame(arr, columns=cols, index=X.index)

        if self.method == "ordinal":
            arr = self.encoder.transform(X)
            # Keep same column names
            return pd.DataFrame(arr, columns=X.columns, index=X.index)

        if self.method == "label":
            # LabelEncoder operates on single column
            return pd.Series(self.encoder.transform(X.iloc[:,0]), index=X.index, name=X.columns[0])
