import pandas as pd
import numpy as np
from scipy.stats import skew

class SmartVisualizer:
    """
    Automatically selects visualizations based on dataset characteristics:
    - Numeric features: histogram, boxplot
        - Skewed features → highlight distribution
    - Categorical features: countplots
        - High cardinality → top N categories
    - Correlation heatmap if numeric features > 1
    """
    def __init__(self, max_categories=10, skew_threshold=1.0):
        self.max_categories = max_categories
        self.skew_threshold = skew_threshold

    def recommend(self, df, target_column=None):
        recommended = []

        # Numeric features
        numeric_features = df.select_dtypes(include="number").columns.tolist()
        if target_column and target_column in numeric_features:
            numeric_features.remove(target_column)

        for col in numeric_features:
            col_skew = skew(df[col].dropna())
            # Always histogram + boxplot
            recommended.append({
                "feature": col,
                "type": "numeric",
                "plots": ["histogram", "boxplot"],
                "skew": col_skew,
                "skewed": abs(col_skew) > self.skew_threshold
            })

        # Categorical features
        categorical_features = df.select_dtypes(exclude="number").columns.tolist()
        if target_column and target_column in categorical_features:
            categorical_features.remove(target_column)

        for col in categorical_features:
            n_unique = df[col].nunique()
            plots = ["countplot"]
            if n_unique > self.max_categories:
                plots.append(f"top_{self.max_categories}_categories")
            recommended.append({
                "feature": col,
                "type": "categorical",
                "plots": plots,
                "n_unique": n_unique
            })

        # Correlation heatmap if multiple numeric features
        if len(numeric_features) > 1:
            recommended.append({
                "feature": "correlation",
                "type": "numeric_summary",
                "plots": ["correlation_heatmap"]
            })

        return recommended
