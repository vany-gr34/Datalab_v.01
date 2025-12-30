"""
Raw data storage management for ingested datasets.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import json
import hashlib
from datetime import datetime


class RawDataStorage:
    """
    Manages raw data storage with directory structure and integrity checks.
    """

    def __init__(self, base_path: str = "storage/raw"):
        """
        Initialize the raw data storage manager.

        Args:
            base_path: Base directory for raw data storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def store_raw_data(self, dataset_id: str, data, file_format: str = 'csv',
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store raw data exactly as received, without transformation.

        Args:
            dataset_id: Unique identifier for the dataset
            data: Raw data to store (DataFrame, dict, etc.)
            file_format: Format to store the data in
            metadata: Optional metadata to store alongside the data

        Returns:
            Dictionary with storage information
        """
        # Create dataset directory
        dataset_path = self.base_path / dataset_id
        dataset_path.mkdir(exist_ok=True)

        storage_info = {
            'dataset_id': dataset_id,
            'storage_path': str(dataset_path),
            'stored_at': datetime.now().isoformat(),
            'file_format': file_format
        }

        # Store the raw data
        if file_format == 'csv':
            data_file = dataset_path / 'data.csv'
            data.to_csv(data_file, index=False)
            storage_info['data_file'] = str(data_file)
            storage_info['file_size'] = data_file.stat().st_size

        elif file_format == 'json':
            data_file = dataset_path / 'data.json'
            if hasattr(data, 'to_dict'):  # DataFrame
                data.to_json(data_file, orient='records', indent=2)
            else:  # dict or list
                with open(data_file, 'w') as f:
                    json.dump(data, f, indent=2)
            storage_info['data_file'] = str(data_file)
            storage_info['file_size'] = data_file.stat().st_size

        elif file_format == 'parquet':
            data_file = dataset_path / 'data.parquet'
            data.to_parquet(data_file, index=False)
            storage_info['data_file'] = str(data_file)
            storage_info['file_size'] = data_file.stat().st_size

        else:
            raise ValueError(f"Unsupported storage format: {file_format}")

        # Generate and store checksum
        checksum = self._generate_checksum(data_file)
        storage_info['checksum'] = checksum

        # Store metadata if provided
        if metadata:
            metadata_file = dataset_path / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            storage_info['metadata_file'] = str(metadata_file)

        # Save storage info
        info_file = dataset_path / 'storage_info.json'
        with open(info_file, 'w') as f:
            json.dump(storage_info, f, indent=2, default=str)

        return storage_info

    def retrieve_raw_data(self, dataset_id: str, file_format: str = 'csv'):
        """
        Retrieve raw data from storage.

        Args:
            dataset_id: Unique identifier of the dataset
            file_format: Expected format of the stored data

        Returns:
            Path to the data file, or None if not found
        """
        dataset_path = self.base_path / dataset_id
        if not dataset_path.exists():
            return None

        if file_format == 'csv':
            data_file = dataset_path / 'data.csv'
        elif file_format == 'json':
            data_file = dataset_path / 'data.json'
        elif file_format == 'parquet':
            data_file = dataset_path / 'data.parquet'
        else:
            return None

        return data_file if data_file.exists() else None

    def verify_integrity(self, dataset_id: str) -> bool:
        """
        Verify the integrity of stored data using checksums.

        Args:
            dataset_id: Unique identifier of the dataset

        Returns:
            True if data is intact, False otherwise
        """
        dataset_path = self.base_path / dataset_id
        info_file = dataset_path / 'storage_info.json'

        if not info_file.exists():
            return False

        # Load storage info
        with open(info_file, 'r') as f:
            storage_info = json.load(f)

        # Check if data file exists
        data_file = Path(storage_info.get('data_file', ''))
        if not data_file.exists():
            return False

        # Verify checksum
        current_checksum = self._generate_checksum(data_file)
        stored_checksum = storage_info.get('checksum')

        return current_checksum == stored_checksum

    def list_stored_datasets(self) -> list:
        """
        List all datasets currently stored.

        Returns:
            List of dataset IDs
        """
        if not self.base_path.exists():
            return []

        return [d.name for d in self.base_path.iterdir() if d.is_dir()]

    def delete_dataset(self, dataset_id: str) -> bool:
        """
        Delete a stored dataset.

        Args:
            dataset_id: Unique identifier of the dataset to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        dataset_path = self.base_path / dataset_id
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
            return True
        return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        total_size = 0
        dataset_count = 0

        for dataset_path in self.base_path.iterdir():
            if dataset_path.is_dir():
                dataset_count += 1
                for file_path in dataset_path.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size

        return {
            'total_datasets': dataset_count,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'storage_path': str(self.base_path)
        }

    def _generate_checksum(self, file_path: Path) -> str:
        """Generate MD5 checksum for a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def archive_dataset(self, dataset_id: str, archive_path: Optional[str] = None) -> Optional[str]:
        """
        Archive a dataset to a compressed format.

        Args:
            dataset_id: Unique identifier of the dataset
            archive_path: Optional path for the archive file

        Returns:
            Path to the created archive, or None if archiving failed
        """
        dataset_path = self.base_path / dataset_id
        if not dataset_path.exists():
            return None

        if archive_path is None:
            archive_path = self.base_path / f"{dataset_id}.zip"

        try:
            shutil.make_archive(str(archive_path).replace('.zip', ''), 'zip', dataset_path)
            return str(archive_path)
        except Exception:
            return None
