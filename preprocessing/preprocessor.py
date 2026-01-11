import pandas as pd
import logging
from preprocessing.scaler import NumericScaler
from preprocessing.encoders import CategoricalEncoder
from preprocessing.missing_values import MissingValueHandler
from preprocessing.outliers import OutlierHandler
import joblib

logger = logging.getLogger(__name__)



class Preprocessor:
    """
    Modular Preprocessor for numeric and categorical features.
    Supports skipping preprocessing completely.
    """

    def __init__(self,
                 numeric_transformers=None,
                 categorical_transformers=None,
                 missing_value_handler=None,
                 outlier_handler=None,
                 skip_preprocessing=False,
                 numeric_columns_to_scale=None,
                 categorical_columns_to_encode=None,
                 columns_to_handle_missing=None,
                 columns_to_handle_outliers=None):
        self.numeric_transformers = numeric_transformers or []
        self.categorical_transformers = categorical_transformers or []
        self.missing_value_handler = missing_value_handler
        self.outlier_handler = outlier_handler
        self.skip_preprocessing = skip_preprocessing

        self.numeric_features = []
        self.categorical_features = []

        # User-specified columns
        self.numeric_columns_to_scale = numeric_columns_to_scale or []
        self.categorical_columns_to_encode = categorical_columns_to_encode or []
        self.columns_to_handle_missing = columns_to_handle_missing or []
        self.columns_to_handle_outliers = columns_to_handle_outliers or []

    def detect_feature_types(self, df, target_column=None):
        # Use user-specified columns if provided, otherwise auto-detect
        if self.numeric_columns_to_scale:
            self.numeric_features = [col for col in self.numeric_columns_to_scale if col in df.columns]
        else:
            numeric = df.select_dtypes(include="number").columns.tolist()
            if target_column and target_column in numeric:
                numeric.remove(target_column)
            self.numeric_features = numeric

        if self.categorical_columns_to_encode:
            self.categorical_features = [col for col in self.categorical_columns_to_encode if col in df.columns]
        else:
            categorical = df.select_dtypes(exclude="number").columns.tolist()
            if target_column and target_column in categorical:
                categorical.remove(target_column)
            self.categorical_features = categorical

    def fit(self, df, target_column=None):
        if self.skip_preprocessing:
            return  # do nothing
        # normalize column names
        df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]

        # detect features after normalization
        self.detect_feature_types(df, target_column=target_column)

        # missing values
        if self.missing_value_handler:
            if self.columns_to_handle_missing:
                # Fit on specified columns
                self.missing_value_handler.fit(df[self.columns_to_handle_missing])
                df[self.columns_to_handle_missing] = self.missing_value_handler.transform(df[self.columns_to_handle_missing])
            else:
                self.missing_value_handler.fit(df)
                df = self.missing_value_handler.transform(df)

        # outliers
        if self.outlier_handler:
            self.outlier_handler.fit(df)
            # do not modify df on fit; actual handling in transform

        # fit numeric transformers
        for transformer in self.numeric_transformers:
            if self.numeric_features:
                transformer.fit(df[self.numeric_features])

        # fit categorical transformers
        for transformer in self.categorical_transformers:
            if self.categorical_features:
                transformer.fit(df[self.categorical_features])

        self.df_columns = df.columns.tolist()

    def transform(self, df):
        if self.skip_preprocessing:
            return df  # return raw dataframe
        df_transformed = df.copy()

        # normalize column names to match fit
        df_transformed.columns = [str(c).strip().lower().replace(' ', '_') for c in df_transformed.columns]

        if self.missing_value_handler:
            df_transformed = self.missing_value_handler.transform(df_transformed)

        # outlier handling
        if self.outlier_handler:
            df_transformed = self.outlier_handler.transform(df_transformed)

        # numeric transforms
        for transformer in self.numeric_transformers:
            if self.numeric_features:
                vals = transformer.transform(df_transformed[self.numeric_features])
                # keep DataFrame shape
                if hasattr(vals, 'shape') and vals.ndim == 2:
                    df_transformed.loc[:, self.numeric_features] = vals
                else:
                    df_transformed.loc[:, self.numeric_features] = vals

        # categorical transforms (may expand columns)
        for transformer in self.categorical_transformers:
            if self.categorical_features:
                transformed = transformer.transform(df_transformed[self.categorical_features])
                if isinstance(transformed, pd.DataFrame):
                    # drop original categorical cols and concat
                    df_transformed = df_transformed.drop(columns=self.categorical_features)
                    df_transformed = pd.concat([df_transformed, transformed.reset_index(drop=True)], axis=1)
                elif isinstance(transformed, pd.Series):
                    df_transformed.loc[:, transformed.name] = transformed
                else:
                    # numpy array with same shape
                    df_transformed.loc[:, self.categorical_features] = transformed

        return df_transformed

    def save_pipeline(self, path):
        """Serialize the preprocessor (transformers and handlers) to disk."""
        to_save = {
            'numeric_transformers': self.numeric_transformers,
            'categorical_transformers': self.categorical_transformers,
            'missing_value_handler': self.missing_value_handler,
            'outlier_handler': self.outlier_handler,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'numeric_columns_to_scale': self.numeric_columns_to_scale,
            'categorical_columns_to_encode': self.categorical_columns_to_encode,
            'columns_to_handle_missing': self.columns_to_handle_missing,
            'columns_to_handle_outliers': self.columns_to_handle_outliers,
        }
        joblib.dump(to_save, path)

    @staticmethod
    def load_pipeline(path):
        return joblib.load(path)

    def fit_transform(self, df, target_column=None):
        if self.skip_preprocessing:
            return df  # return raw dataframe
        self.fit(df, target_column=target_column)
        return self.transform(df)
