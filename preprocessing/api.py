"""
Preprocessing API - Main entry point for data preprocessing workflows.

Orchestrates the full preprocessing pipeline:
1. Missing value handling
2. Outlier detection/handling
3. Encoding categorical features
4. Scaling numeric features
5. Metadata tracking and versioning

Integrates with visualization layer for before/after comparison.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from preprocessing import (
    Preprocessor, MissingValueHandler, OutlierHandler,
    CategoricalEncoder, NumericScaler
)
from preprocessing.metadata import PreprocessingMetadata, PreprocessingPipeline
from helpers.visualization_exporter import (
    StatisticalSummaryExporter, ComparisonExporter, VisualizationExporter
)

logger = logging.getLogger(__name__)


class PreprocessingAPI:
    """High-level API for data preprocessing with tracking and versioning."""

    def __init__(self):
        self.pipelines = {}  # pipeline_id -> PreprocessingPipeline
        self.processing_history = []

    def create_preprocessing_config(self,
                                   handle_missing: str = 'mean',
                                   handle_outliers: Optional[str] = None,
                                   scale_numeric: str = 'standard',
                                   encode_categorical: str = 'onehot',
                                   numeric_columns_to_scale: Optional[list] = None,
                                   categorical_columns_to_encode: Optional[list] = None,
                                   columns_to_handle_missing: Optional[list] = None,
                                   columns_to_handle_outliers: Optional[list] = None) -> Dict[str, Any]:
        """Create a preprocessing configuration.

        Args:
            handle_missing: 'mean'|'median'|'most_frequent'|'delete'|None
            handle_outliers: 'iqr'|'zscore'|None
            scale_numeric: 'standard'|'minmax'|'robust'|None
            encode_categorical: 'onehot'|'ordinal'|'label'|None
            numeric_columns_to_scale: List of numeric columns to scale. If None, auto-detects.
            categorical_columns_to_encode: List of categorical columns to encode. If None, auto-detects.
            columns_to_handle_missing: List of columns to handle missing values. If None, applies to all.
            columns_to_handle_outliers: List of columns to handle outliers. If None, applies to all.

        Returns:
            Configuration dict for passing to preprocess()
        """
        return {
            'handle_missing': handle_missing,
            'handle_outliers': handle_outliers,
            'scale_numeric': scale_numeric,
            'encode_categorical': encode_categorical,
            'numeric_columns_to_scale': numeric_columns_to_scale,
            'categorical_columns_to_encode': categorical_columns_to_encode,
            'columns_to_handle_missing': columns_to_handle_missing,
            'columns_to_handle_outliers': columns_to_handle_outliers,
        }

    def preprocess(self,
                   df: pd.DataFrame,
                   config: Dict[str, Any],
                   dataset_id: str,
                   target_column: Optional[str] = None,
                   is_training: bool = True) -> Tuple[pd.DataFrame, PreprocessingMetadata]:
        """Execute full preprocessing pipeline.

        Args:
            df: Input DataFrame
            config: Preprocessing configuration
            dataset_id: Unique dataset identifier
            target_column: Name of target column to exclude from certain operations
            is_training: If True, fit transformers; if False, use existing transformers

        Returns:
            (processed_df, metadata)
        """
        df_original = df.copy()

        # Initialize metadata
        metadata = PreprocessingMetadata(dataset_id, config)
        metadata.set_input_data_info(df)

        try:
            # 1. Missing value handling
            missing_handler = None
            if config.get('handle_missing') and config['handle_missing'] != 'none':
                missing_handler = MissingValueHandler(
                    strategy='impute' if config['handle_missing'] != 'delete' else 'delete',
                    fill_strategy=config['handle_missing'] if config['handle_missing'] != 'delete' else 'mean'
                )
                if is_training:
                    missing_handler.fit(df)
                df = missing_handler.transform(df)
                metadata.log_step('missing_value_handling', 'MissingValueHandler',
                                 {'strategy': config['handle_missing']},
                                 input_shape=df_original.shape, output_shape=df.shape)

            # 2. Outlier handling
            outlier_handler = None
            if config.get('handle_outliers') and config['handle_outliers'] != 'none':
                outlier_handler = OutlierHandler(
                    method=config['handle_outliers'],
                    mode='remove'
                )
                if is_training:
                    outlier_handler.fit(df)
                df = outlier_handler.transform(df)
                metadata.log_step('outlier_handling', 'OutlierHandler',
                                 {'method': config['handle_outliers']},
                                 input_shape=df.shape, output_shape=df.shape)

            # Detect feature types after initial cleaning
            preprocessor = Preprocessor(
                numeric_columns_to_scale=config.get('numeric_columns_to_scale'),
                categorical_columns_to_encode=config.get('categorical_columns_to_encode'),
                columns_to_handle_missing=config.get('columns_to_handle_missing'),
                columns_to_handle_outliers=config.get('columns_to_handle_outliers')
            )
            preprocessor.detect_feature_types(df, target_column=target_column)

            # 3. Categorical encoding
            numeric_transformers = []
            categorical_transformers = []

            if config.get('encode_categorical') and config['encode_categorical'] != 'none':
                encoder = CategoricalEncoder(method=config['encode_categorical'])
                categorical_transformers.append(encoder)
                if is_training and preprocessor.categorical_features:
                    encoder.fit(df[preprocessor.categorical_features])
                metadata.log_step('categorical_encoding', 'CategoricalEncoder',
                                 {'method': config['encode_categorical']})

            # 4. Numeric scaling
            if config.get('scale_numeric') and config['scale_numeric'] != 'none':
                scaler = NumericScaler(method=config['scale_numeric'])
                numeric_transformers.append(scaler)
                if is_training and preprocessor.numeric_features:
                    scaler.fit(df[preprocessor.numeric_features])
                metadata.log_step('numeric_scaling', 'NumericScaler',
                                 {'method': config['scale_numeric']})

            # 5. Apply all transformers
            preprocessor.numeric_transformers = numeric_transformers
            preprocessor.categorical_transformers = categorical_transformers
            preprocessor.missing_value_handler = missing_handler
            preprocessor.outlier_handler = outlier_handler

            df_processed = preprocessor.fit_transform(df, target_column=target_column) if is_training else preprocessor.transform(df)

            # Update metadata with final output info
            metadata.set_output_data_info(df_processed)

            # Store pipeline for reproducibility
            pipeline = PreprocessingPipeline(dataset_id, preprocessor, metadata)
            self.pipelines[dataset_id] = pipeline

            # Log processing history
            self.processing_history.append({
                'dataset_id': dataset_id,
                'timestamp': metadata.timestamp,
                'input_shape': df_original.shape,
                'output_shape': df_processed.shape,
                'config': config,
            })

            logger.info(f"Preprocessing complete for {dataset_id}. Input: {df_original.shape}, Output: {df_processed.shape}")
            return df_processed, metadata

        except Exception as e:
            logger.error(f"Preprocessing failed for {dataset_id}: {str(e)}")
            raise

    def get_preprocessing_summary(self, dataset_id: str) -> Dict[str, Any]:
        """Get summary of preprocessing applied to a dataset."""
        if dataset_id not in self.pipelines:
            return {'error': f'No pipeline found for {dataset_id}'}

        pipeline = self.pipelines[dataset_id]
        return pipeline.metadata.get_summary()

    def compare_datasets(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, Any]:
        """Generate a comparison report between original and processed datasets."""
        return ComparisonExporter.compare_datasets(df_before, df_after)

    def export_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Export comprehensive statistics for a dataset."""
        return {
            'numeric_summary': StatisticalSummaryExporter.numeric_summary(df),
            'categorical_summary': StatisticalSummaryExporter.categorical_summary(df),
            'missing_values': StatisticalSummaryExporter.missing_values_summary(df),
            'correlation': StatisticalSummaryExporter.correlation_summary(df),
            'overview': StatisticalSummaryExporter.dataset_overview(df),
        }
