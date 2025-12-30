"""
Standardized dataset object for the ingestion layer.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
import uuid


@dataclass
class Dataset:
    """
    Standardized dataset object containing data and metadata.

    This class encapsulates a pandas DataFrame along with comprehensive
    metadata about the dataset's origin, processing, and characteristics.
    """

    data: pd.DataFrame
    metadata: Dict[str, Any]
    id: Optional[str] = None
    source_type: Optional[str] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        """Initialize default values after dataclass initialization."""
        if self.id is None:
            self.id = str(uuid.uuid4())

        if self.created_at is None:
            self.created_at = datetime.now()

        # Update metadata with dataset ID
        if self.metadata:
            self.metadata['dataset_id'] = self.id
            self.metadata['created_at'] = self.created_at.isoformat()

    @property
    def shape(self) -> tuple:
        """Return the shape of the dataset (rows, columns)."""
        return self.data.shape

    @property
    def columns(self) -> list:
        """Return the list of column names."""
        return list(self.data.columns)

    @property
    def dtypes(self) -> pd.Series:
        """Return the data types of each column."""
        return self.data.dtypes

    @property
    def num_rows(self) -> int:
        """Return the number of rows in the dataset."""
        return len(self.data)

    @property
    def num_columns(self) -> int:
        """Return the number of columns in the dataset."""
        return len(self.data.columns)

    @property
    def memory_usage(self) -> str:
        """Return a human-readable string of memory usage."""
        usage_bytes = self.data.memory_usage(deep=True).sum()
        for unit in ['B', 'KB', 'MB', 'GB']:
            if usage_bytes < 1024.0:
                return ".1f"
            usage_bytes /= 1024.0
        return ".1f"

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the dataset including basic statistics.

        Returns:
            Dictionary containing dataset summary information
        """
        summary = {
            'dataset_id': self.id,
            'source_type': self.source_type,
            'shape': self.shape,
            'columns': self.columns,
            'dtypes': {col: str(dtype) for col, dtype in self.dtypes.items()},
            'memory_usage': self.memory_usage,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'metadata': self.metadata
        }

        # Add basic statistics for numeric columns
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary['numeric_stats'] = self.data[numeric_cols].describe().to_dict()

        # Add info about missing values
        summary['missing_values'] = self.data.isnull().sum().to_dict()

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the dataset to a dictionary representation.

        Returns:
            Dictionary containing all dataset information
        """
        return {
            'id': self.id,
            'data': self.data.to_dict('records'),
            'metadata': self.metadata,
            'source_type': self.source_type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'shape': self.shape,
            'columns': self.columns
        }

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'Dataset':
        """
        Create a Dataset instance from a dictionary.

        Args:
            data_dict: Dictionary containing dataset information

        Returns:
            Dataset instance
        """
        data = pd.DataFrame(data_dict['data'])
        metadata = data_dict.get('metadata', {})
        dataset_id = data_dict.get('id')
        source_type = data_dict.get('source_type')
        created_at_str = data_dict.get('created_at')

        created_at = None
        if created_at_str:
            created_at = datetime.fromisoformat(created_at_str)

        return cls(
            data=data,
            metadata=metadata,
            id=dataset_id,
            source_type=source_type,
            created_at=created_at
        )

    def copy(self) -> 'Dataset':
        """
        Create a copy of the dataset.

        Returns:
            New Dataset instance with copied data and metadata
        """
        return Dataset(
            data=self.data.copy(),
            metadata=self.metadata.copy() if self.metadata else {},
            id=str(uuid.uuid4()),  # Generate new ID for copy
            source_type=self.source_type,
            created_at=datetime.now()
        )

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return f"Dataset(id='{self.id}', shape={self.shape}, source='{self.source_type}')"
