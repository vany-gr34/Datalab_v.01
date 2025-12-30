"""
Visualization exporters for backend-to-frontend data serialization.

Supports PNG, JSON, and interactive chart data formats for
seamless integration with React frontend.
"""

import io
import json
import base64
import logging
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class VisualizationExporter:
    """Export matplotlib figures to multiple formats."""

    @staticmethod
    def fig_to_base64_png(fig) -> str:
        """Convert matplotlib figure to base64-encoded PNG."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return f'data:image/png;base64,{encoded}'

    @staticmethod
    def fig_to_png_file(fig, output_path: str) -> str:
        """Save matplotlib figure to PNG file. Returns file path."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format='png', dpi=100, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")
        return output_path

    @staticmethod
    def figs_to_json(figs_dict: Dict[str, Any]) -> str:
        """Convert dict of figures to JSON with base64-encoded images."""
        result = {}
        for name, fig in figs_dict.items():
            if hasattr(fig, 'savefig'):
                # matplotlib figure
                result[name] = {
                    'type': 'matplotlib',
                    'image': VisualizationExporter.fig_to_base64_png(fig)
                }
            else:
                # assume already serializable
                result[name] = fig
        return json.dumps(result)


class StatisticalSummaryExporter:
    """Export dataset statistics and summary data for visualization."""

    @staticmethod
    def numeric_summary(df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Any]:
        """Generate summary statistics for numeric columns."""
        if columns is None:
            columns = df.select_dtypes(include='number').columns.tolist()

        summary = {}
        for col in columns:
            if col in df.columns:
                s = df[col].describe().to_dict()
                s['skewness'] = float(df[col].skew())
                s['kurtosis'] = float(df[col].kurtosis())
                summary[col] = s
        return summary

    @staticmethod
    def categorical_summary(df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Any]:
        """Generate summary for categorical columns (value counts, etc.)."""
        if columns is None:
            columns = df.select_dtypes(exclude='number').columns.tolist()

        summary = {}
        for col in columns:
            if col in df.columns:
                vc = df[col].value_counts().to_dict()
                summary[col] = {
                    'value_counts': vc,
                    'unique_count': df[col].nunique(),
                    'missing_count': df[col].isnull().sum()
                }
        return summary

    @staticmethod
    def missing_values_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate missing values report."""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        return {
            'total_missing': missing.sum(),
            'columns': {
                col: {
                    'count': int(missing[col]),
                    'percentage': float(missing_pct[col])
                }
                for col in df.columns if missing[col] > 0
            }
        }

    @staticmethod
    def correlation_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate correlation matrix for numeric columns."""
        numeric = df.select_dtypes(include='number')
        if numeric.shape[1] < 2:
            return {}
        corr = numeric.corr()
        return {
            'correlation_matrix': corr.to_dict(),
            'shape': corr.shape,
        }

    @staticmethod
    def dataset_overview(df: pd.DataFrame, before_preprocess: pd.DataFrame = None) -> Dict[str, Any]:
        """Generate comprehensive dataset overview."""
        overview = {
            'current_shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            'row_count': len(df),
            'column_count': len(df.columns),
        }

        if before_preprocess is not None:
            overview['before_shape'] = before_preprocess.shape
            overview['rows_removed'] = before_preprocess.shape[0] - df.shape[0]
            overview['columns_added'] = df.shape[1] - before_preprocess.shape[1]

        return overview


class ComparisonExporter:
    """Export before/after preprocessing comparisons."""

    @staticmethod
    def compare_datasets(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, Any]:
        """Generate a comparison report between two datasets."""
        return {
            'before': {
                'shape': df_before.shape,
                'columns': list(df_before.columns),
                'dtypes_count': df_before.dtypes.value_counts().to_dict(),
                'missing_values': StatisticalSummaryExporter.missing_values_summary(df_before),
            },
            'after': {
                'shape': df_after.shape,
                'columns': list(df_after.columns),
                'dtypes_count': df_after.dtypes.value_counts().to_dict(),
                'missing_values': StatisticalSummaryExporter.missing_values_summary(df_after),
            },
            'transformation_summary': {
                'rows_removed': df_before.shape[0] - df_after.shape[0],
                'rows_kept': df_after.shape[0],
                'columns_added': df_after.shape[1] - df_before.shape[1],
                'columns_removed': df_before.shape[1] - df_after.shape[1],
            }
        }

    @staticmethod
    def compare_distributions(df_before: pd.DataFrame, df_after: pd.DataFrame,
                             columns: List[str] = None) -> Dict[str, Any]:
        """Compare distributions of numeric columns before/after."""
        if columns is None:
            columns = df_before.select_dtypes(include='number').columns.tolist()

        comparisons = {}
        for col in columns:
            if col in df_before.columns and col in df_after.columns:
                comparisons[col] = {
                    'before': {
                        'mean': float(df_before[col].mean()),
                        'std': float(df_before[col].std()),
                        'min': float(df_before[col].min()),
                        'max': float(df_before[col].max()),
                    },
                    'after': {
                        'mean': float(df_after[col].mean()),
                        'std': float(df_after[col].std()),
                        'min': float(df_after[col].min()),
                        'max': float(df_after[col].max()),
                    }
                }
        return comparisons
