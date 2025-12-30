"""
Preprocessing metadata and versioning system for reproducibility.

Tracks all preprocessing steps, parameters, and output statistics.
Enables pipeline reuse for training/inference consistency.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd

logger = logging.getLogger(__name__)


class PreprocessingMetadata:
    """Track preprocessing operations and transformations."""

    def __init__(self, dataset_id: str, source_config: Dict[str, Any] = None):
        self.dataset_id = dataset_id
        self.source_config = source_config or {}
        self.timestamp = datetime.now().isoformat()
        self.steps: List[Dict[str, Any]] = []
        self.input_data_info = {}
        self.output_data_info = {}

    def log_step(self, step_name: str, transformer_type: str, params: Dict[str, Any],
                 input_shape: tuple = None, output_shape: tuple = None, notes: str = None):
        """Log a preprocessing step with details."""
        self.steps.append({
            'step_index': len(self.steps),
            'step_name': step_name,
            'transformer_type': transformer_type,
            'params': params,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'notes': notes,
            'timestamp': datetime.now().isoformat()
        })
        logger.debug(f"Logged preprocessing step: {step_name}")

    def set_input_data_info(self, df: pd.DataFrame):
        """Capture input data statistics."""
        self.input_data_info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        }

    def set_output_data_info(self, df: pd.DataFrame):
        """Capture output data statistics."""
        self.output_data_info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of preprocessing operations."""
        return {
            'dataset_id': self.dataset_id,
            'timestamp': self.timestamp,
            'num_steps': len(self.steps),
            'input_shape': self.input_data_info.get('shape'),
            'output_shape': self.output_data_info.get('shape'),
            'columns_added': len(self.output_data_info.get('columns', [])) - len(self.input_data_info.get('columns', [])),
            'steps': self.steps,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'dataset_id': self.dataset_id,
            'timestamp': self.timestamp,
            'source_config': self.source_config,
            'input_data_info': self.input_data_info,
            'output_data_info': self.output_data_info,
            'steps': self.steps,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreprocessingMetadata':
        """Reconstruct from dictionary."""
        obj = cls(data['dataset_id'], data.get('source_config', {}))
        obj.timestamp = data.get('timestamp')
        obj.input_data_info = data.get('input_data_info', {})
        obj.output_data_info = data.get('output_data_info', {})
        obj.steps = data.get('steps', [])
        return obj


class PreprocessingPipeline:
    """Encapsulate a reusable preprocessing pipeline for consistency across train/inference."""

    def __init__(self, pipeline_id: str, preprocessor, metadata: PreprocessingMetadata = None):
        """pipeline_id: unique identifier for versioning
        preprocessor: Preprocessor instance
        metadata: PreprocessingMetadata instance
        """
        self.pipeline_id = pipeline_id
        self.preprocessor = preprocessor
        self.metadata = metadata or PreprocessingMetadata(pipeline_id)

    def get_config(self) -> Dict[str, Any]:
        """Return pipeline configuration for reproducibility."""
        return {
            'pipeline_id': self.pipeline_id,
            'skip_preprocessing': self.preprocessor.skip_preprocessing,
            'numeric_transformers': [t.__class__.__name__ for t in self.preprocessor.numeric_transformers],
            'categorical_transformers': [t.__class__.__name__ for t in self.preprocessor.categorical_transformers],
            'missing_value_handler': self.preprocessor.missing_value_handler.__class__.__name__
                                     if self.preprocessor.missing_value_handler else None,
            'outlier_handler': self.preprocessor.outlier_handler.__class__.__name__
                               if self.preprocessor.outlier_handler else None,
        }

    def apply(self, df: pd.DataFrame, is_training: bool = True):
        """Apply the pipeline, optionally logging metadata."""
        self.metadata.set_input_data_info(df)

        if is_training:
            result = self.preprocessor.fit_transform(df)
        else:
            result = self.preprocessor.transform(df)

        self.metadata.set_output_data_info(result)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize pipeline for persistence."""
        return {
            'pipeline_id': self.pipeline_id,
            'config': self.get_config(),
            'metadata': self.metadata.to_dict(),
        }
