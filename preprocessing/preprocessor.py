import pandas as pd
from preprocessing.scaler import NumericScaler
from preprocessing.encoders import CategoricalEncoder
from preprocessing.missing_values import MissingValueHandler

class Preprocessor:
    """
    Modular Preprocessor for numeric and categorical features.
    Supports skipping preprocessing completely.
    """

    def __init__(self, 
                 numeric_transformers=None, 
                 categorical_transformers=None, 
                 missing_value_handler=None,
                 skip_preprocessing=False):
        self.numeric_transformers = numeric_transformers or []
        self.categorical_transformers = categorical_transformers or []
        self.missing_value_handler = missing_value_handler
        self.skip_preprocessing = skip_preprocessing

        self.numeric_features = []
        self.categorical_features = []

    def detect_feature_types(self, df, target_column=None):
        numeric = df.select_dtypes(include="number").columns.tolist()
        categorical = df.select_dtypes(exclude="number").columns.tolist()
        if target_column:
            if target_column in numeric:
                numeric.remove(target_column)
            if target_column in categorical:
                categorical.remove(target_column)
        self.numeric_features = numeric
        self.categorical_features = categorical

    def fit(self, df):
        if self.skip_preprocessing:
            return  # do nothing

        if self.missing_value_handler:
            self.missing_value_handler.fit(df)
            df = self.missing_value_handler.transform(df)

        for transformer in self.numeric_transformers:
            if self.numeric_features:
                transformer.fit(df[self.numeric_features])

        for transformer in self.categorical_transformers:
            if self.categorical_features:
                transformer.fit(df[self.categorical_features])

        self.df_columns = df.columns.tolist()

    def transform(self, df):
        if self.skip_preprocessing:
            return df  # return raw dataframe

        df_transformed = df.copy()

        if self.missing_value_handler:
            df_transformed = self.missing_value_handler.transform(df_transformed)

        for transformer in self.numeric_transformers:
            if self.numeric_features:
                df_transformed[self.numeric_features] = transformer.transform(df_transformed[self.numeric_features])

        for transformer in self.categorical_transformers:
            if self.categorical_features:
                df_transformed[self.categorical_features] = transformer.transform(df_transformed[self.categorical_features])

        return df_transformed

    def fit_transform(self, df):
        if self.skip_preprocessing:
            return df  # return raw dataframe
        self.fit(df)
        return self.transform(df)
