"""
Abstract base class for all data ingestors.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class BaseIngestor(ABC):
    """
    Abstract base class for data ingestion modules.

    Each ingestor must implement the ingest method to handle
    data collection from a specific source type.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ingestor with configuration.

        Args:
            config: Dictionary containing source-specific configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def ingest(self) -> 'Dataset':
        """
        Ingest data from the configured source.

        Returns:
            Dataset: A standardized dataset object containing the data and metadata

        Raises:
            IngestionError: If ingestion fails
        """
        pass

    def validate_config(self) -> bool:
        """
        Validate the configuration parameters.

        Returns:
            bool: True if config is valid, False otherwise
        """
        return True

    def _log_ingestion_start(self, source_info: str):
        """Log the start of ingestion process."""
        self.logger.info(f"Starting ingestion from {source_info}")

    def _log_ingestion_success(self, source_info: str, record_count: int):
        """Log successful ingestion."""
        self.logger.info(f"Successfully ingested {record_count} records from {source_info}")

    def _log_ingestion_error(self, source_info: str, error: Exception):
        """Log ingestion error."""
        self.logger.error(f"Failed to ingest from {source_info}: {str(error)}")

    async def ingest_async(self) -> 'Dataset':
        """
        Asynchronous version of ingest method.

        Returns:
            Dataset: A standardized dataset object containing the data and metadata

        Raises:
            IngestionError: If ingestion fails
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.ingest)

    def ingest_with_timeout(self, timeout_seconds: int = 300) -> 'Dataset':
        """
        Ingest data with a timeout to prevent hanging operations.

        Args:
            timeout_seconds: Maximum time to wait for ingestion (default: 5 minutes)

        Returns:
            Dataset: A standardized dataset object

        Raises:
            IngestionError: If ingestion fails or times out
        """
        result = [None]
        exception = [None]

        def run_ingestion():
            try:
                result[0] = self.ingest()
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=run_ingestion)
        thread.start()
        thread.join(timeout_seconds)

        if thread.is_alive():
            raise IngestionError(f"Ingestion timed out after {timeout_seconds} seconds")

        if exception[0]:
            raise exception[0]

        return result[0]


class IngestionError(Exception):
    """Custom exception for ingestion-related errors."""
    pass
