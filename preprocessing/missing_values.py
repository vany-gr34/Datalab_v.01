from sklearn.impute import SimpleImputer
from preprocessing.BaseTransformer import BaseTransformer
import pandas as pd
import numpy as np

class MissingValueHandler(BaseTransformer):

    def __init__(self, strategy="impute", fill_strategy="mean"):
        self.strategy = strategy
        self.fill_strategy = fill_strategy
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.numeric_columns = None
        self.categorical_columns = None
        self.indicator_columns = []
        
    def fit(self, X):
        # Identify numeric and categorical columns
        if isinstance(X, pd.DataFrame):
            self.numeric_columns = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
            self.categorical_columns = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        if self.strategy in ["impute", "impute_indicator"]:
            # Handle numeric columns
            if self.numeric_columns:
                numeric_data = X[self.numeric_columns] if isinstance(X, pd.DataFrame) else X
                self.numeric_imputer = SimpleImputer(strategy=self.fill_strategy)
                self.numeric_imputer.fit(numeric_data)
            
            # Handle categorical columns - can't use mean for categorical
            if self.categorical_columns:
                categorical_data = X[self.categorical_columns] if isinstance(X, pd.DataFrame) else X
                # For categorical data, use most_frequent if fill_strategy is mean
                if self.fill_strategy == "mean":
                    cat_strategy = "most_frequent"
                else:
                    cat_strategy = self.fill_strategy
                
                # For constant strategy with categorical, use 'missing' as default
                if cat_strategy == "constant":
                    self.categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')
                else:
                    self.categorical_imputer = SimpleImputer(strategy=cat_strategy)
                
                self.categorical_imputer.fit(categorical_data)
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X_transformed = X.copy()
        
        if self.strategy == "delete":
            # Drop rows with any missing values
            original_shape = X_transformed.shape
            X_transformed = X_transformed.dropna()
            st.info(f"Deleted {original_shape[0] - X_transformed.shape[0]} rows with missing values")
            return X_transformed
        
        elif self.strategy in ["impute", "impute_indicator"]:
            # Handle missing value indicators first if needed
            if self.strategy == "impute_indicator":
                # Add indicator columns for missing values
                for col in X_transformed.columns:
                    if X_transformed[col].isnull().any():
                        indicator_col = col + "_missing"
                        X_transformed[indicator_col] = X_transformed[col].isnull().astype(int)
                        self.indicator_columns.append(indicator_col)
            
            # Impute numeric columns
            if self.numeric_columns and self.numeric_imputer:
                # Filter only columns that exist in current data
                existing_numeric_cols = [col for col in self.numeric_columns if col in X_transformed.columns]
                if existing_numeric_cols:
                    X_transformed[existing_numeric_cols] = self.numeric_imputer.transform(
                        X_transformed[existing_numeric_cols]
                    )
            
            # Impute categorical columns
            if self.categorical_columns and self.categorical_imputer:
                # Filter only columns that exist in current data
                existing_categorical_cols = [col for col in self.categorical_columns if col in X_transformed.columns]
                if existing_categorical_cols:
                    imputed_values = self.categorical_imputer.transform(
                        X_transformed[existing_categorical_cols]
                    )
                    # Convert back to DataFrame to preserve column names
                    X_transformed[existing_categorical_cols] = pd.DataFrame(
                        imputed_values, 
                        columns=existing_categorical_cols,
                        index=X_transformed.index
                    )
            
            return X_transformed
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")