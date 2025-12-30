"""
User upload ingestor for handling file uploads through backend API.
"""

import pandas as pd
from typing import Dict, Any, Optional
from io import BytesIO, StringIO
import base64
from .base_ingestor import BaseIngestor, IngestionError
from .dataset import Dataset
from .metadata import MetadataManager


class UserUploadIngestor(BaseIngestor):
    """
    Ingestor for handling user file uploads through backend API (Flask/FastAPI).
    """

    SUPPORTED_FORMATS = {
        'csv': 'csv',
        'xlsx': 'excel',
        'xls': 'excel',
        'json': 'json',
        'parquet': 'parquet'
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the user upload ingestor.

        Args:
            config: Configuration dictionary containing:
                - file_content: Base64 encoded file content or raw bytes
                - file_name: Original filename
                - file_type: File type (auto-detected if not provided)
                - content_type: MIME type of the file
                - max_file_size: Maximum allowed file size in bytes
        """
        super().__init__(config)
        self.file_content = config.get('file_content')
        self.file_name = config.get('file_name', 'uploaded_file')
        self.file_type = config.get('file_type')
        self.content_type = config.get('content_type')
        self.max_file_size = config.get('max_file_size', 100 * 1024 * 1024)  # 100MB default
        self.metadata_manager = MetadataManager()

        if not self.file_content:
            raise ValueError("file_content is required in config")

    def validate_config(self) -> bool:
        """Validate the upload configuration."""
        # Check file size
        if isinstance(self.file_content, str):
            # Base64 encoded content
            content_size = len(self.file_content) * 3 / 4  # Approximate decoded size
        else:
            # Raw bytes
            content_size = len(self.file_content)

        if content_size > self.max_file_size:
            self.logger.error(f"File size {content_size} exceeds maximum allowed size {self.max_file_size}")
            return False

        # Auto-detect file type if not provided
        if not self.file_type:
            self.file_type = self._detect_file_type_from_name()

        if self.file_type not in self.SUPPORTED_FORMATS:
            self.logger.error(f"Unsupported file type: {self.file_type}")
            return False

        return True

    def ingest(self) -> Dataset:
        """
        Ingest data from the uploaded file content.

        Returns:
            Dataset: Standardized dataset object

        Raises:
            IngestionError: If ingestion fails
        """
        try:
            self._log_ingestion_start(f"upload: {self.file_name}")

            if not self.validate_config():
                raise IngestionError("Invalid upload configuration")

            # Decode and read the file content
            data = self._read_uploaded_content()

            # Validate and clean data
            data = self._validate_and_clean_data(data)

            # Generate metadata
            data_info = {
                'data': data,
                'file_name': self.file_name,
                'file_type': self.file_type,
                'content_type': self.content_type
            }

            metadata = self.metadata_manager.generate_metadata(
                dataset_id=f"upload_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                source_type='upload',
                source_config=self.config,
                data_info=data_info
            )

            dataset = Dataset(data, metadata)

            self._log_ingestion_success(f"upload: {self.file_name}", len(data))

            return dataset

        except Exception as e:
            self._log_ingestion_error(f"upload: {self.file_name}", e)
            raise IngestionError(f"Failed to ingest uploaded file: {str(e)}")

    def _detect_file_type_from_name(self) -> str:
        """Detect file type from filename."""
        if '.' in self.file_name:
            extension = self.file_name.split('.')[-1].lower()
            return self.SUPPORTED_FORMATS.get(extension, 'unknown')
        return 'unknown'

    def _read_uploaded_content(self) -> pd.DataFrame:
        """Read the uploaded file content into a DataFrame."""
        # Decode base64 if necessary
        if isinstance(self.file_content, str):
            # Assume base64 encoded
            try:
                decoded_content = base64.b64decode(self.file_content)
            except Exception as e:
                raise IngestionError(f"Failed to decode base64 content: {str(e)}")
        else:
            # Raw bytes
            decoded_content = self.file_content

        # Convert to file-like object
        file_obj = BytesIO(decoded_content)

        # Read based on file type
        if self.file_type == 'csv':
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1']
            for encoding in encodings:
                try:
                    file_obj.seek(0)
                    content_str = file_obj.read().decode(encoding)
                    return pd.read_csv(StringIO(content_str))
                except (UnicodeDecodeError, pd.errors.EmptyDataError):
                    continue
            raise IngestionError("Unable to decode CSV file with supported encodings")

        elif self.file_type in ['xlsx', 'xls']:
            try:
                return pd.read_excel(file_obj)
            except Exception as e:
                raise IngestionError(f"Failed to read Excel file: {str(e)}")

        elif self.file_type == 'json':
            try:
                file_obj.seek(0)
                content_str = file_obj.read().decode('utf-8')
                return pd.read_json(StringIO(content_str))
            except Exception as e:
                raise IngestionError(f"Failed to read JSON file: {str(e)}")

        elif self.file_type == 'parquet':
            try:
                return pd.read_parquet(file_obj)
            except Exception as e:
                raise IngestionError(f"Failed to read Parquet file: {str(e)}")

        else:
            raise IngestionError(f"Unsupported file type: {self.file_type}")

    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and perform basic cleaning on the uploaded data.

        Args:
            data: Raw DataFrame from upload

        Returns:
            Cleaned DataFrame
        """
        # Check for empty data
        if data.empty:
            raise IngestionError("Uploaded file contains no data")

        # Check for malicious content (basic check)
        if self._contains_malicious_content(data):
            raise IngestionError("File contains potentially malicious content")

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

    def _contains_malicious_content(self, data: pd.DataFrame) -> bool:
        """
        Basic check for potentially malicious content in uploaded files.
        This is a simple implementation - in production, use more sophisticated checks.
        """
        # Check for suspicious column names
        suspicious_patterns = ['script', 'javascript', 'vbscript', 'onload', 'onerror', '<script>']

        for col in data.columns:
            col_str = str(col).lower()
            for pattern in suspicious_patterns:
                if pattern in col_str:
                    return True

        # Check for suspicious values in string columns
        for col in data.select_dtypes(include=['object', 'string']).columns:
            sample_values = data[col].dropna().head(10).astype(str).str.lower()
            for value in sample_values:
                for pattern in suspicious_patterns:
                    if pattern in value:
                        return True

        return False
