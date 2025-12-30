from preprocessing.BaseTransformer import BaseTransformer
import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class OutlierHandler(BaseTransformer):
    """Detect and optionally handle outliers using IQR or Z-score methods.

    mode: 'detect'|'cap'|'remove'
    method: 'iqr'|'zscore'
    threshold: for zscore, numeric value; for iqr, multiplier (1.5 default)
    """
    def __init__(self, method='iqr', mode='detect', threshold=1.5):
        self.method = method
        self.mode = mode
        self.threshold = threshold
        self.cols_ = None

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            self.cols_ = X.select_dtypes(include='number').columns.tolist()
        else:
            self.cols_ = []

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        df = X.copy()

        if not self.cols_:
            self.fit(df)

        if self.method == 'iqr':
            for col in self.cols_:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.threshold * iqr
                upper = q3 + self.threshold * iqr
                mask_lower = df[col] < lower
                mask_upper = df[col] > upper
                if self.mode == 'detect':
                    logger.debug(f"{col}: {mask_lower.sum()+mask_upper.sum()} outliers detected by IQR")
                elif self.mode == 'cap':
                    df.loc[mask_lower, col] = lower
                    df.loc[mask_upper, col] = upper
                elif self.mode == 'remove':
                    df = df[~(mask_lower | mask_upper)]

        elif self.method == 'zscore':
            for col in self.cols_:
                z = np.abs(stats.zscore(df[col].dropna()))
                # stats.zscore returns array aligned to non-na values; build mask
                mask = pd.Series(False, index=df.index)
                non_na_idx = df[col].dropna().index
                mask.loc[non_na_idx] = z > self.threshold
                if self.mode == 'detect':
                    logger.debug(f"{col}: {mask.sum()} outliers detected by zscore")
                elif self.mode == 'cap':
                    # cap at nearest non-outlier boundary
                    non_out = df.loc[~mask, col]
                    if not non_out.empty:
                        lower = non_out.min()
                        upper = non_out.max()
                        df.loc[mask & (df[col] < lower), col] = lower
                        df.loc[mask & (df[col] > upper), col] = upper
                elif self.mode == 'remove':
                    df = df[~mask]

        else:
            raise ValueError(f"Unknown outlier method: {self.method}")

        return df
