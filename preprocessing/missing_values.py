from sklearn.impute import SimpleImputer
from preprocessing.BaseTransformer import BaseTransformer
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MissingValueHandler(BaseTransformer):

	def __init__(self, strategy="impute", fill_strategy="mean"):
		"""strategy: 'delete', 'impute', or 'impute_indicator'
		fill_strategy: for numeric - 'mean'|'median'|'most_frequent'|'constant'
		"""
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
			self.numeric_columns = X.select_dtypes(include=['number']).columns.tolist()
			self.categorical_columns = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

		if self.strategy in ["impute", "impute_indicator"]:
			# Numeric imputer
			if self.numeric_columns:
				numeric_data = X[self.numeric_columns]
				self.numeric_imputer = SimpleImputer(strategy=self.fill_strategy)
				self.numeric_imputer.fit(numeric_data)

			# Categorical imputer
			if self.categorical_columns:
				# Map mean -> most_frequent for categorical
				cat_strategy = 'most_frequent' if self.fill_strategy == 'mean' else self.fill_strategy
				if cat_strategy == 'constant':
					self.categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')
				else:
					self.categorical_imputer = SimpleImputer(strategy=cat_strategy)
				self.categorical_imputer.fit(X[self.categorical_columns])

	def transform(self, X):
		if not isinstance(X, pd.DataFrame):
			X = pd.DataFrame(X)

		X_transformed = X.copy()

		if self.strategy == "delete":
			original_shape = X_transformed.shape
			X_transformed = X_transformed.dropna()
			logger.info(f"Deleted {original_shape[0] - X_transformed.shape[0]} rows with missing values")
			return X_transformed

		elif self.strategy in ["impute", "impute_indicator"]:
			if self.strategy == "impute_indicator":
				for col in X_transformed.columns:
					if X_transformed[col].isnull().any():
						indicator_col = f"{col}_missing"
						X_transformed[indicator_col] = X_transformed[col].isnull().astype(int)
						self.indicator_columns.append(indicator_col)

			# Impute numeric
			if self.numeric_columns and self.numeric_imputer:
				existing_numeric_cols = [c for c in self.numeric_columns if c in X_transformed.columns]
				if existing_numeric_cols:
					X_transformed.loc[:, existing_numeric_cols] = self.numeric_imputer.transform(
						X_transformed[existing_numeric_cols]
					)

			# Impute categorical
			if self.categorical_columns and self.categorical_imputer:
				existing_cat_cols = [c for c in self.categorical_columns if c in X_transformed.columns]
				if existing_cat_cols:
					imputed = self.categorical_imputer.transform(X_transformed[existing_cat_cols])
					X_transformed.loc[:, existing_cat_cols] = pd.DataFrame(imputed, columns=existing_cat_cols, index=X_transformed.index)

			return X_transformed

		else:
			raise ValueError(f"Unknown strategy: {self.strategy}")

