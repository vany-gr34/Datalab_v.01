"""
File ingestor for handling local file uploads and parsing.
"""

import pandas as pd
import os
from typing import Dict, Any, Optional
import json
from pathlib import Path
import chardet
from .base_ingestor import BaseIngestor, IngestionError
from .dataset import Dataset
from .metadata import MetadataManager


class FileIngestor(BaseIngestor):
    """
    Ingestor for local files including CSV, Excel, JSON, and Parquet formats.
    """

    SUPPORTED_FORMATS = {
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.json': 'json',
        '.parquet': 'parquet'
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the file ingestor.

        Args:
            config: Configuration dictionary containing:
                - file_path: Path to the file to ingest
                - file_type: Optional file type override
                - encoding: Optional encoding specification
                - chunk_size: Optional chunk size for large files
        """
        super().__init__(config)
        self.file_path = config.get('file_path')
        self.file_type = config.get('file_type')
        self.encoding = config.get('encoding', 'auto')
        self.chunk_size = config.get('chunk_size')
        self.metadata_manager = MetadataManager()

        if not self.file_path:
            raise ValueError("file_path is required in config")

    def validate_config(self) -> bool:
        """Validate the configuration."""
        if not os.path.exists(self.file_path):
            self.logger.error(f"File does not exist: {self.file_path}")
            return False

        file_ext = Path(self.file_path).suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS and not self.file_type:
            self.logger.error(f"Unsupported file format: {file_ext}")
            return False

        return True

    def ingest(self) -> Dataset:
        """
        Ingest data from the specified file.

        Returns:
            Dataset: Standardized dataset object

        Raises:
            IngestionError: If ingestion fails
        """
        try:
            self._log_ingestion_start(f"file: {self.file_path}")

            if not self.validate_config():
                raise IngestionError("Invalid configuration")

            # Auto-detect file type if not specified
            if not self.file_type:
                self.file_type = self._detect_file_type()

            # Read the file based on type
            if self.chunk_size and self.file_type in ['csv', 'parquet']:
                data = self._read_file_chunked()
            else:
                data = self._read_file()

            # Validate and clean data
            data = self._validate_and_clean_data(data)

            # Generate metadata
            data_info = {
                'data': data,
                'file_size': os.path.getsize(self.file_path),
                'file_type': self.file_type
            }

            metadata = self.metadata_manager.generate_metadata(
                dataset_id=f"file_{Path(self.file_path).stem}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                source_type='file',
                source_config=self.config,
                data_info=data_info
            )

            dataset = Dataset(data, metadata)

            self._log_ingestion_success(f"file: {self.file_path}", len(data))

            return dataset

        except Exception as e:
            self._log_ingestion_error(f"file: {self.file_path}", e)
            raise IngestionError(f"Failed to ingest file {self.file_path}: {str(e)}")

    def _detect_file_type(self) -> str:
        """Auto-detect file type based on extension."""
        file_ext = Path(self.file_path).suffix.lower()
        return self.SUPPORTED_FORMATS.get(file_ext, 'unknown')

    def _detect_encoding(self) -> str:
        """Detect file encoding."""
        if self.encoding != 'auto':
            return self.encoding

        with open(self.file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
            result = chardet.detect(raw_data)
            return result.get('encoding', 'utf-8')

    def _read_file(self) -> pd.DataFrame:
        """Read file based on its type."""
        encoding = self._detect_encoding()

        if self.file_type == 'csv':
            return pd.read_csv(self.file_path, encoding=encoding)
        elif self.file_type == 'excel':
            return pd.read_excel(self.file_path)
        elif self.file_type == 'json':
            return pd.read_json(self.file_path, encoding=encoding)
        elif self.file_type == 'parquet':
            return pd.read_parquet(self.file_path)
        else:
            raise IngestionError(f"Unsupported file type: {self.file_type}")

    def _read_file_chunked(self) -> pd.DataFrame:
        """Read large files in chunks."""
        if self.file_type == 'csv':
            chunks = []
            for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
                chunks.append(chunk)
            return pd.concat(chunks, ignore_index=True)
        elif self.file_type == 'parquet':
            # Parquet files can be read in chunks using pyarrow
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(self.file_path)
            chunks = []
            for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
                chunks.append(batch.to_pandas())
            return pd.concat(chunks, ignore_index=True)
        else:
            # Fallback to regular reading
            return self._read_file()

    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and perform basic cleaning on the data.

        Args:
            data: Raw DataFrame from file

        Returns:
            Cleaned DataFrame
        """
        # Check for empty data
        if data.empty:
            raise IngestionError("File contains no data")

        # Normalize column names
        data.columns = [str(col).strip().lower().replace(' ', '_') for col in data.columns]

        # Basic data type inference
        data = data.convert_dtypes()

        # Log data quality issues
        missing_pct = (data.isnull().sum() / len(data) * 100).round(2)
        if (missing_pct > 50).any():
            self.logger.warning(f"High missing value percentages: {missing_pct[missing_pct > 50].to_dict()}")

        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            self.logger.info(f"Found {duplicate_count} duplicate rows")

        return data
