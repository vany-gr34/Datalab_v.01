"""
Ingestion registry and unified interface for data ingestion.
"""

from typing import Dict, Any, Optional
import time
from .base_ingestor import IngestionError
from .file_ingestor import FileIngestor
from .db_ingestor import DatabaseIngestor
from .api_ingestor import APIIngestor
from .user_upload_ingestor import UserUploadIngestor
from .dataset import Dataset
from .storage import RawDataStorage
from .metadata import MetadataManager
from .logger import get_ingestion_logger


class IngestionRegistry:
    """
    Registry for managing pluggable ingestion modules and providing unified interface.
    """

    def __init__(self):
        """Initialize the ingestion registry."""
        self.ingestors = {
            'file': FileIngestor,
            'db': DatabaseIngestor,
            'api': APIIngestor,
            'upload': UserUploadIngestor
        }

        self.storage = RawDataStorage()
        self.metadata_manager = MetadataManager()
        # Use the IngestionLogger wrapper for custom logging methods
        self.ingestion_logger = get_ingestion_logger()
        # Keep a standard logger for typical logging calls
        self.logger = self.ingestion_logger.get_logger(__name__)

    def register_ingestor(self, source_type: str, ingestor_class):
        """
        Register a new ingestor class.

        Args:
            source_type: Type identifier for the ingestor
            ingestor_class: Ingestor class to register
        """
        self.ingestors[source_type] = ingestor_class
        self.logger.info(f"Registered ingestor for source type: {source_type}")

    def get_ingestor(self, source_type: str):
        """
        Get an ingestor class by source type.

        Args:
            source_type: Type of data source

        Returns:
            Ingestor class

        Raises:
            ValueError: If source type is not supported
        """
        if source_type not in self.ingestors:
            available_types = list(self.ingestors.keys())
            raise ValueError(f"Unsupported source type: {source_type}. Available types: {available_types}")

        return self.ingestors[source_type]

    def ingest(self, source_type: str, source_config: Dict[str, Any],
               store_raw: bool = True, generate_metadata: bool = True) -> Dataset:
        """
        Unified ingestion interface.

        Args:
            source_type: Type of data source ('file', 'db', 'api', 'upload')
            source_config: Configuration specific to the source type
            store_raw: Whether to store raw data
            generate_metadata: Whether to generate metadata

        Returns:
            Dataset: Standardized dataset object

        Raises:
            IngestionError: If ingestion fails
        """
        start_time = time.time()

        try:
            # use ingestion_logger for higher-level ingestion logging
            self.ingestion_logger.log_ingestion_attempt(source_type, str(source_config.get('location', source_config.get('url', 'unknown'))), source_config)

            # Get the appropriate ingestor
            ingestor_class = self.get_ingestor(source_type)

            # Create ingestor instance
            ingestor = ingestor_class(source_config)

            # Perform ingestion
            dataset = ingestor.ingest()

            # Store raw data if requested
            if store_raw:
                storage_info = self.storage.store_raw_data(
                    dataset.id,
                    dataset.data,
                    file_format='csv',  # Default to CSV for storage
                    metadata=dataset.metadata if generate_metadata else None
                )
                dataset.metadata['storage_info'] = storage_info

            # Update metadata if requested
            if generate_metadata:
                self.metadata_manager.save_metadata(dataset.metadata)

            # Log success
            duration = time.time() - start_time
            self.ingestion_logger.log_ingestion_success(source_type, str(source_config.get('location', source_config.get('url', 'unknown'))), len(dataset.data), dataset.id)
            self.ingestion_logger.log_performance_metrics(f"{source_type}_ingestion", duration, len(dataset.data))

            return dataset

        except Exception as e:
            duration = time.time() - start_time
            self.ingestion_logger.log_ingestion_error(source_type, str(source_config.get('location', source_config.get('url', 'unknown'))), e)
            raise IngestionError(f"Ingestion failed for {source_type}: {str(e)}") from e

    def list_supported_sources(self) -> list:
        """
        List all supported source types.

        Returns:
            List of supported source type strings
        """
        return list(self.ingestors.keys())

    def get_ingestion_history(self) -> list:
        """
        Get ingestion history from metadata.

        Returns:
            List of ingested datasets with metadata
        """
        datasets = self.metadata_manager.list_datasets()
        history = []

        for dataset_id in datasets:
            metadata = self.metadata_manager.load_metadata(dataset_id)
            if metadata:
                history.append({
                    'dataset_id': dataset_id,
                    'source_type': metadata.get('source_type'),
                    'ingestion_timestamp': metadata.get('ingestion_timestamp'),
                    'num_rows': metadata.get('data_info', {}).get('num_rows'),
                    'num_columns': metadata.get('data_info', {}).get('num_columns'),
                    'status': metadata.get('status', 'unknown')
                })

        return sorted(history, key=lambda x: x.get('ingestion_timestamp', ''), reverse=True)


# Global registry instance
_ingestion_registry = None


def get_ingestion_registry() -> IngestionRegistry:
    """Get the global ingestion registry instance."""
    global _ingestion_registry
    if _ingestion_registry is None:
        _ingestion_registry = IngestionRegistry()
    return _ingestion_registry


def ingest(source_type: str, source_config: Dict[str, Any],
           store_raw: bool = True, generate_metadata: bool = True) -> Dataset:
    """
    Unified data ingestion function.

    This is the main entry point for data ingestion in the DataLab system.

    Args:
        source_type: Type of data source ('file', 'db', 'api', 'upload')
        source_config: Configuration dictionary for the source
        store_raw: Whether to store raw data (default: True)
        generate_metadata: Whether to generate metadata (default: True)

    Returns:
        Dataset: Standardized dataset object containing the data and metadata

    Raises:
        IngestionError: If ingestion fails

    Examples:
        # Ingest from local CSV file
        dataset = ingest('file', {'file_path': 'data.csv'})

        # Ingest from database
        dataset = ingest('db', {
            'db_type': 'postgresql',
            'host': 'localhost',
            'database': 'mydb',
            'table': 'customers',
            'username': 'user',
            'password': 'pass'
        })

        # Ingest from REST API
        dataset = ingest('api', {
            'url': 'https://api.example.com/data',
            'auth_type': 'api_key',
            'auth_config': {'key_name': 'X-API-Key', 'key_value': 'your-key'}
        })

        # Ingest uploaded file
        dataset = ingest('upload', {
            'file_content': base64_content,
            'file_name': 'uploaded.csv',
            'file_type': 'csv'
        })
    """
    registry = get_ingestion_registry()
    return registry.ingest(source_type, source_config, store_raw, generate_metadata)
