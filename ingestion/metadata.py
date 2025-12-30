"""
Metadata generation and management for ingested datasets.
"""

import json
from typing import Dict, Any, Optional
from datetime import datetime
import os
import hashlib


class MetadataManager:
    """
    Manages metadata generation, storage, and retrieval for datasets.
    """

    def __init__(self, storage_path: str = "storage/metadata"):
        """
        Initialize the metadata manager.

        Args:
            storage_path: Path where metadata files will be stored
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

    def generate_metadata(self, dataset_id: str, source_type: str, source_config: Dict[str, Any],
                         data_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive metadata for a dataset.

        Args:
            dataset_id: Unique identifier for the dataset
            source_type: Type of data source (file, db, api, etc.)
            source_config: Configuration used for ingestion
            data_info: Information about the ingested data

        Returns:
            Dictionary containing all metadata
        """
        metadata = {
            'dataset_id': dataset_id,
            'source_type': source_type,
            'source_location': source_config.get('location', source_config.get('url', 'unknown')),
            'ingestion_timestamp': datetime.now().isoformat(),
            'ingestion_config': source_config,
            'data_info': data_info,
            'version': '1.0',
            'status': 'active'
        }

        # Add data quality metrics
        if 'data' in data_info:
            data = data_info['data']
            metadata.update(self._generate_data_quality_metrics(data))

        return metadata

    def _generate_data_quality_metrics(self, data) -> Dict[str, Any]:
        """Generate data quality metrics."""
        metrics = {
            'num_rows': len(data),
            'num_columns': len(data.columns),
            'data_types': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'missing_values': data.isnull().sum().to_dict(),
            'duplicate_rows': int(data.duplicated().sum()),
            'constant_columns': [col for col in data.columns if data[col].nunique() == 1],
            'potential_targets': self._identify_potential_targets(data)
        }

        # Calculate file size approximation
        metrics['file_size'] = data.memory_usage(deep=True).sum()

        # Generate checksum
        data_str = data.to_csv(index=False)
        metrics['checksum'] = hashlib.md5(data_str.encode()).hexdigest()

        return metrics

    def _identify_potential_targets(self, data) -> list:
        """Identify columns that could be potential target variables."""
        potential_targets = []

        for col in data.columns:
            unique_ratio = data[col].nunique() / len(data)
            if unique_ratio < 0.5:  # Less than 50% unique values
                potential_targets.append(col)

        return potential_targets

    def save_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Save metadata to a JSON file.

        Args:
            metadata: Metadata dictionary to save

        Returns:
            Path to the saved metadata file
        """
        dataset_id = metadata['dataset_id']
        filename = f"{dataset_id}_metadata.json"
        filepath = os.path.join(self.storage_path, filename)

        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        return filepath

    def load_metadata(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Load metadata for a specific dataset.

        Args:
            dataset_id: Unique identifier of the dataset

        Returns:
            Metadata dictionary if found, None otherwise
        """
        filename = f"{dataset_id}_metadata.json"
        filepath = os.path.join(self.storage_path, filename)

        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)

        return None

    def list_datasets(self) -> list:
        """
        List all available datasets based on metadata files.

        Returns:
            List of dataset IDs
        """
        if not os.path.exists(self.storage_path):
            return []

        datasets = []
        for filename in os.listdir(self.storage_path):
            if filename.endswith('_metadata.json'):
                dataset_id = filename.replace('_metadata.json', '')
                datasets.append(dataset_id)

        return datasets

    def update_metadata_status(self, dataset_id: str, status: str) -> bool:
        """
        Update the status of a dataset.

        Args:
            dataset_id: Unique identifier of the dataset
            status: New status (active, archived, deleted, etc.)

        Returns:
            True if update was successful, False otherwise
        """
        metadata = self.load_metadata(dataset_id)
        if metadata:
            metadata['status'] = status
            metadata['last_updated'] = datetime.now().isoformat()
            self.save_metadata(metadata)
            return True

        return False
